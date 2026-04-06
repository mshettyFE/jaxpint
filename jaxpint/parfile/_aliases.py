"""Parameter name aliases: alternate name → canonical PINT name."""

from __future__ import annotations

# Extracted from PINT parameter definitions across all component files.
# Keys are uppercase alternate names; values are the canonical names used
# internally by JaxPINT and PINT.
ALIASES: dict[str, str] = {
    # Astrometry
    "RA": "RAJ",
    "DEC": "DECJ",
    "LAMBDA": "ELONG",
    "BETA": "ELAT",
    "PMLAMBDA": "PMELONG",
    "PMBETA": "PMELAT",
    # Binary common
    "E": "ECC",
    "X": "A1",
    "XDOT": "A1DOT",
    "FB": "FB0",
    # Binary DD
    "DTHETA": "DTH",
    "VARSIGMA": "STIGMA",
    "STIG": "STIGMA",
    # Noise
    "T2EFAC": "EFAC",
    "TNEF": "EFAC",
    "T2EQUAD": "EQUAD",
    "TNECORR": "ECORR",
    # Solar wind
    "NE1AU": "NE_SW",
    "SOLARN0": "NE_SW",
    # Timing model metadata
    "PSRJ": "PSR",
    "PSRB": "PSR",
    "CLK": "CLOCK",
}

# Prefix aliases: alternate prefix → canonical prefix.
# For indexed parameters like EXPEP_1 → EXPDIPEPOCH_1.
PREFIX_ALIASES: dict[str, str] = {
    "EXPEP_": "EXPDIPEPOCH_",
    "EXPPH_": "EXPDIPAMP_",
    "EXPINDEX_": "EXPDIPIDX_",
    "EXPTAU_": "EXPDIPTAU_",
}


def resolve_alias(name: str) -> str:
    """Resolve a parameter name to its canonical form.

    Checks the direct alias map first, then prefix aliases for indexed
    parameters.
    """
    upper = name.upper()

    # Direct alias
    if upper in ALIASES:
        return ALIASES[upper]

    # Prefix alias: check if the name starts with any known prefix alias
    for alt_prefix, canon_prefix in PREFIX_ALIASES.items():
        if upper.startswith(alt_prefix):
            suffix = upper[len(alt_prefix):]
            return canon_prefix + suffix

    return upper
