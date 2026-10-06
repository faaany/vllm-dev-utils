# Loaded automatically by Python at startup when this directory is on PYTHONPATH.
# If VLLM_STEP_TIMERS=1, installs step_timers right after the trigger module is
# imported (default: vLLM's worker module, which exists only in engine processes).
import os
import sys

# Keep the system sitecustomize behaviour (Ubuntu: apport crash handler).
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

if os.environ.get("VLLM_STEP_TIMERS") == "1":
    import importlib.abc
    import importlib.util

    _TRIGGER = os.environ.get("VLLM_STEP_TIMERS_TRIGGER", "vllm.v1.worker.gpu_worker")

    class _InstallAfterImport(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TRIGGER:
                return None
            sys.meta_path.remove(self)  # one-shot; let the normal finders locate it
            spec = importlib.util.find_spec(name)
            exec_module = spec.loader.exec_module

            def exec_then_install(module):
                exec_module(module)
                import step_timers
                step_timers.install()

            spec.loader.exec_module = exec_then_install
            return spec

    sys.meta_path.insert(0, _InstallAfterImport())
