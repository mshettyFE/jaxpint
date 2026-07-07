"""Sphinx configuration for JaxPINT documentation."""

import sys
from pathlib import Path

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# -- Project information ------------------------------------------------------
project = "JaxPINT"
author = "JaxPINT Contributors"
release = "0.1.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc settings ---------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"

# -- Autosummary settings -----------------------------------------------------
autosummary_generate = True

# -- Cross-reference strictness -----------------------------------------------
nitpicky = True
nitpick_ignore: list[tuple[str, str]] = [
    # TypeVar in map_pulsars' generic signature (Callable[[PulsarRecord], T]
    # -> Iterator[T]); autodoc emits it as an unresolvable class reference,
    # both qualified and bare.
    ("py:class", "jaxpint.loaders.nanograv.T"),
    ("py:class", "T"),
    # Docstring shorthand in the frequentist detection/stats modules (rendered
    # for the first time via jaxpint.frequentist's curated re-exports).
    ("py:class", "PRNGKey"),
    ("py:class", "array-like"),
]
# jaxtyping annotations (e.g. ``Float[Array, "n_toas"]``) tokenize into several
# pieces that autodoc emits as separate ``py:class`` refs — none of which
# resolve. The patterns below silence that noise so real broken refs in
# narrative docs stay visible.
nitpick_ignore_regex: list[tuple[str, str]] = [
    # jaxtyping types and shape strings — autodoc tokenizes
    # ``Float[Array, "n_toas"]`` into several pieces, none of which resolve.
    ("py:class", r"jaxtyping\..*"),
    ("py:class", r"Array\s*"),
    ("py:class", r"Float Array\s*"),
    ("py:class", r"'.*"),  # opens of quoted shape strings
    ("py:class", r".*'\s*"),  # closes of quoted shape strings (`` n'`` etc.)
    ("py:class", r".*\]\s*"),  # trailing ``]`` from bracketed shapes
    ("py:class", r"\(.*"),  # ``(n_toas`` etc.
    ("py:class", r".*\)\s*"),  # ``..)`` etc.
    ("py:class", r"shape .*"),
    ("py:class", r"\d+ \* .*"),  # ``2 * n_freqs)``
    # JAX private import path; the public re-export is ``jax.random.PRNGKey``.
    ("py:class", r"jax\._src\..*"),
    ("py:class", r"JAX PRNG key\s*"),
    # Generic Python descriptors that show up bare in NumPy-style docstrings
    ("py:class", r"tuple\s*"),
    ("py:class", r"dict\s*"),
    ("py:class", r"tuple\[.*"),  # ``tuple[str, ...]`` tokenizes as ``tuple[str``
    ("py:class", r"dict\[.*"),
    ("py:class", r"array\s*"),
    ("py:class", r"scalar\s*"),
    ("py:class", r"callable\s*"),
    ("py:class", r"sequence\s*"),
    ("py:class", r"\d-D arrays?\s*"),
    ("py:class", r"same shape as .*"),
    ("py:class", r"optional .*"),
    ("py:class", r"Float\s*"),  # bare jaxtyping Float
    ("py:class", r"Float\[.*"),  # ``Float[Array, ...]`` tokenizes as ``Float[Array``
    ("py:class", r".* array\s*"),  # ``(n_toas, n_basis) array`` etc.
    # Project-internal types referenced by bare name in docstrings
    ("py:class", r"ParameterVector\s*"),
    ("py:class", r"TOAData\s*"),
    ("py:class", r"NoiseModel\s*"),
    ("py:class", r"NoiseComponent\s*"),
    ("py:class", r"TimingModel\s*"),
    ("py:class", r"DualFloat\s*"),
    ("py:class", r".*GlobalParams\s*"),
    ("py:class", r".*PulsarBundle\s*"),
    ("py:class", r".*ParamSpec\s*"),
    ("py:class", r".*RawTOA\s*"),
    # PINT references — PINT does not publish an intersphinx objects.inv
    ("py:class", r"pint\..*"),
    ("py:func", r"pint\..*"),
    ("py:meth", r"pint\..*"),
    # Private (underscore-prefixed) submodule paths
    ("py:class", r"jaxpint\.[\w.]*\._\w+(\.\w+)*"),
    ("py:func", r"jaxpint\.[\w.]*\._\w+(\.\w+)*"),
]

# -- Napoleon settings (NumPy docstrings) -------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# -- Intersphinx (cross-reference external docs) ------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
}

# -- HTML output --------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "JaxPINT"
html_theme_options = {
    "github_url": "https://github.com/mshettyFE/jaxpint",
    "show_toc_level": 2,
}
