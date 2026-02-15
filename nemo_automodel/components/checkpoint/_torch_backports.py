# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime back-ports for old PyTorch versions. Will be deleted in future stable PyTorch versions."""

from __future__ import annotations

import importlib
import logging
import socket
import threading

logger = logging.getLogger(__name__)


def apply_patches() -> None:
    """
    Inject modified modules into an *old* ``torch.distributed.checkpoint``.
    """
    # -----------------------------------------------------------------
    # Ensure SavePlanner provides the _cached_metadata class attribute.
    # This is required by NeMo-Automodel's extended planners but may be
    # missing from older PyTorch versions (< 2.4).  Monkey-patch it here
    # so downstream code can rely on its existence independent of the
    # installed torch release.
    # -----------------------------------------------------------------
    try:
        planner_mod = importlib.import_module("torch.distributed.checkpoint.planner")
        SavePlanner = getattr(planner_mod, "SavePlanner", None)
        if SavePlanner is not None and not hasattr(SavePlanner, "_cached_metadata"):
            # Forward-declare attribute; note we don't import Metadata to
            # avoid circular deps – a forward reference string in the
            # annotation keeps static checkers happy while remaining
            # runtime-safe.
            SavePlanner._cached_metadata = {}

            # Update type annotations dynamically for better type hints
            anns = getattr(SavePlanner, "__annotations__", {})
            anns.setdefault("_cached_metadata", "dict[str, 'Metadata']")
            SavePlanner.__annotations__ = anns  # type: ignore[attr-defined]

            logger.debug("Added missing SavePlanner._cached_metadata back-port")
    except ModuleNotFoundError:
        # planner module unavailable – nothing to patch
        pass


def apply_async_checkpoint_patch() -> None:
    """
    Apply stabilization patch for torch.distributed.checkpoint async process executor.
    This serializes creation of the global background process across concurrent async_save calls.
    """
    try:
        ape_mod = importlib.import_module("torch.distributed.checkpoint._async_process_executor")

        # Idempotent guard
        if getattr(ape_mod, "_NEMO_PATCHED_CREATE_LOCK", False):
            return

        # Global creation lock
        if not hasattr(ape_mod, "_NEMO_CREATE_LOCK"):
            ape_mod._NEMO_CREATE_LOCK = threading.Lock()

        Exec = getattr(ape_mod, "_ProcessBasedAsyncCheckpointExecutor", None)
        if Exec is not None and not hasattr(Exec, "_nemo_orig_execute_save_impl"):
            Exec._nemo_orig_execute_save_impl = Exec._execute_save_impl

            def _nemo_locked_execute_save_impl(*args, **kwargs):
                with ape_mod._NEMO_CREATE_LOCK:
                    return Exec._nemo_orig_execute_save_impl(*args, **kwargs)

            try:
                Exec._execute_save_impl = staticmethod(_nemo_locked_execute_save_impl)
                logger.debug("Applied creation-lock patch to DCP process executor")
            except Exception:
                # Defensive: if staticmethod replacement fails, leave as-is
                logger.debug("Failed to assign locked _execute_save_impl", exc_info=True)

        ape_mod._NEMO_PATCHED_CREATE_LOCK = True
    except ModuleNotFoundError:
        # async_process_executor unavailable – nothing to patch
        pass
    except Exception:
        logger.debug("Unexpected error while applying DCP process executor patch", exc_info=True)


def apply_async_port_validation_patch() -> None:
    """
    Replace ``get_free_port`` in the DCP async process executor module with a
    version that validates the port is available on **all interfaces** (0.0.0.0).

    The upstream implementation binds only to localhost, which can miss ports
    already in use on other interfaces (e.g. by a stale DCP background process).
    Gloo's ``init_process_group`` binds on 0.0.0.0, causing EADDRINUSE.
    """
    try:
        ape_mod = importlib.import_module(
            "torch.distributed.checkpoint._async_process_executor"
        )

        if getattr(ape_mod, "_NEMO_PATCHED_PORT_VALIDATION", False):
            return

        orig_get_free_port = getattr(ape_mod, "get_free_port", None)
        if orig_get_free_port is None:
            return

        def _validated_get_free_port(max_retries: int = 10) -> int:
            """Get a free port and verify it is bindable on 0.0.0.0."""
            for attempt in range(max_retries):
                port = orig_get_free_port()
                try:
                    # Verify the port is free on all interfaces
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("0.0.0.0", port))
                    sock.close()
                    return port
                except OSError:
                    logger.debug(
                        "Port %d busy on 0.0.0.0, retrying (%d/%d)",
                        port,
                        attempt + 1,
                        max_retries,
                    )
            # Last resort: let the OS pick a port on 0.0.0.0 directly
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", 0))
            port = sock.getsockname()[1]
            sock.close()
            logger.debug("All retries exhausted, OS-assigned port %d on 0.0.0.0", port)
            return port

        ape_mod.get_free_port = _validated_get_free_port
        ape_mod._NEMO_PATCHED_PORT_VALIDATION = True
        logger.debug("Applied 0.0.0.0 port validation patch to DCP get_free_port")

    except ModuleNotFoundError:
        pass
    except Exception:
        logger.debug(
            "Unexpected error while applying DCP port validation patch",
            exc_info=True,
        )
