"""Load a standalone bench script by path without turning bench/ into a package.

`bench/` is a collection of standalone experiment scripts: several import torch or
read server-side result directories at module scope. Importing the directory as a
package would therefore be fragile and would defeat the "offline, no GPU" property
of this test suite. Loading a single module by file path keeps the tests honest:
they exercise the exact code that produced the published numbers.
"""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(relpath, name=None):
    """Import ``ROOT/relpath`` as an isolated module and return it."""
    path = ROOT / relpath
    if not path.is_file():
        raise FileNotFoundError(path)
    name = name or ("_legacy_" + path.stem)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
