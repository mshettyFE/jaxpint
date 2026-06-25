"""PINT is an optional dependency.

Verifies (in a subprocess where ``pint`` is blocked from importing) that
``import jaxpint`` and the native pipeline work without PINT, while PINT-backed
symbols raise a clear ``ImportError`` only on access.  Also a static guard that
``import pint`` does not appear in the PINT-free core.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]

# Run in a subprocess so the real PINT installed in the dev env can't mask the
# missing-PINT behaviour.  A meta_path finder raises ImportError for `pint*`.
_BLOCK_PINT = """
import sys
class _NoPint:
    def find_spec(self, name, path=None, target=None):
        if name == "pint" or name.startswith("pint."):
            raise ImportError("pint is blocked for this test")
        return None
sys.meta_path.insert(0, _NoPint())
for m in list(sys.modules):
    if m == "pint" or m.startswith("pint."):
        del sys.modules[m]
"""


def _run(body: str):
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_PINT + body],
        capture_output=True, text=True, cwd=str(_REPO),
    )


def test_import_jaxpint_without_pint():
    r = _run("import jaxpint; print('OK', jaxpint.__name__)")
    assert r.returncode == 0, r.stderr
    assert "OK jaxpint" in r.stdout


def test_native_path_usable_without_pint():
    # The native namespace + a representative native symbol import & are callable.
    r = _run(
        "import jaxpint\n"
        "assert callable(jaxpint.native.get_model_and_toas)\n"
        "assert callable(jaxpint.native_toas_to_jax)\n"
        "assert callable(jaxpint.build_model)\n"
        "assert callable(jaxpint.load_nanograv_pta)  # native loader, PINT-free\n"
        "import jaxpint.clock, jaxpint.tim, jaxpint.par  # PINT-free subsystems\n"
        "print('OK')\n"
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


@pytest.mark.parametrize("symbol", [
    "pint_toas_to_jax", "build_timing_model", "pint_model_to_params",
    "extract_tzr_toa", "params_to_pint_model",
])
def test_pint_backed_symbol_raises_clear_error(symbol):
    r = _run(
        "import jaxpint\n"
        "try:\n"
        f"    jaxpint.{symbol}\n"
        "    print('NO_ERROR')\n"
        "except ImportError as e:\n"
        "    print('IMPORTERROR', 'jaxpint[pint]' in str(e))\n"
    )
    assert r.returncode == 0, r.stderr
    assert "IMPORTERROR True" in r.stdout, r.stdout


def test_no_pint_import_outside_bridge():
    """The PINT-free core must not regrow an `import pint`.

    Allowed locations: jaxpint/bridge/, notebook_utils.py (and only inside
    functions there).
    """
    allowed = {
        _REPO / "jaxpint" / "bridge",
        _REPO / "jaxpint" / "notebook_utils.py",
    }

    def _is_allowed(p: pathlib.Path) -> bool:
        return any(p == a or (a.is_dir() and a in p.parents) for a in allowed)

    offenders = []
    for py in (_REPO / "jaxpint").rglob("*.py"):
        if _is_allowed(py):
            continue
        for i, line in enumerate(py.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("import pint") or s.startswith("from pint"):
                offenders.append(f"{py.relative_to(_REPO)}:{i}: {s}")
    assert not offenders, "PINT imported in PINT-free core:\n" + "\n".join(offenders)


def test_components_do_not_import_par_spec():
    """Components must not import ``jaxpint.par.spec`` (would risk a cycle).

    ``par.spec`` lazily imports the component classes to aggregate their
    ``PARAMS``; the no-cycle invariant relies on the reverse edge never
    existing -- components import ``ParamDecl`` from ``jaxpint.components`` and
    never import ``par.spec``.  Guard the component modules (and ``model.py``,
    which ``par.spec`` also imports) against an ``import`` of it.  Only ``import``
    statements are inspected, so ``:mod:`jaxpint.par.spec``` docstring mentions
    are ignored.
    """
    files = [_REPO / "jaxpint" / "components.py", _REPO / "jaxpint" / "model.py"]
    for pkg in ("phase", "delay", "binary", "noise"):
        files.extend((_REPO / "jaxpint" / pkg).rglob("*.py"))

    offenders = []
    for py in files:
        for i, line in enumerate(py.read_text().splitlines(), 1):
            s = line.strip()
            if not (s.startswith("import ") or s.startswith("from ")):
                continue
            if "par.spec" in s or "par import spec" in s:
                offenders.append(f"{py.relative_to(_REPO)}:{i}: {s}")
    assert not offenders, (
        "component module imports jaxpint.par.spec (cycle risk):\n"
        + "\n".join(offenders)
    )


def test_pyproject_pint_is_optional():
    import tomllib

    with open(_REPO / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)
    project = cfg["project"]
    hard = " ".join(project["dependencies"])
    extras = project["optional-dependencies"]

    # pint-pulsar is an extra, not a hard dependency
    assert "pint-pulsar" not in hard, hard
    assert any("pint-pulsar" in d for d in extras.get("pint", [])), extras
    # native-path deps promoted to hard
    for d in ("astropy", "pyerfa", "jplephem"):
        assert d in hard, f"{d} missing from hard dependencies"
