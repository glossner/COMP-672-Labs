"""Check the local CPU lab environment; no downloads or network calls."""
import importlib
import sys

if not (3, 11) <= sys.version_info[:2] <= (3, 13):
    raise RuntimeError("Use Python 3.11–3.13.")
print("Python", sys.version.split()[0])
for name in ["numpy", "scipy", "sklearn", "torch", "matplotlib"]:
    module = importlib.import_module(name)
    print(name, module.__version__)
print("CPU laboratory environment ready.")
