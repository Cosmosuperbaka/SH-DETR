"""Central path configuration for the RSC-DETR code base.

Historically every script hard-coded absolute paths pointing at one person's
home directory on the experiment server. Those paths are resolved here instead,
so the repository can be relocated or shared without editing nineteen files.

Resolution order for every entry:

1. the corresponding environment variable, if it is set to a non-empty value;
2. otherwise a default derived from ``RSCDETR_WORKSPACE`` -- which itself
   defaults to the current user's home directory.

Environment variables
---------------------

==========================  ================================================
``RSCDETR_WORKSPACE``        Root that holds all experiment trees.
                            Default: the current user's home directory.
``RSCDETR_ROOT``             Main RSC-DETR working repository.
                            Default: ``$RSCDETR_WORKSPACE/rsc-detr``
``RSCDETR_DATASETS``         Dataset root containing ``VEDAI/`` and ``M3FD/``.
                            Default: ``$RSCDETR_WORKSPACE/vedai_data/datasets``
``RSCDETR_CFT``              CFT comparison checkout.
                            Default: ``$RSCDETR_WORKSPACE/CFT``
``RSCDETR_LCAFNET``          LCAFNet comparison checkout.
                            Default: ``$RSCDETR_WORKSPACE/LCAFNet``
``RSCDETR_MSOD``             Multispectral-object-detection checkout.
                            Default: ``$RSCDETR_WORKSPACE/multispectral-object-detection``
``RSCDETR_PAPER``            Unpacked paper sources (default figure target).
                            Default: ``$RSCDETR_ROOT/RSC_DETR``
``RSCDETR_OUT_QUAL``         Output root for qualitative figures.
                            Default: ``$RSCDETR_ROOT/out_qual``
==========================  ================================================

Run ``python3 rscdetr_paths.py`` to print every path that would be used.

Usage from a script anywhere in the tree::

    import sys
    from pathlib import Path

    for _parent in Path(__file__).resolve().parents:
        if (_parent / "rscdetr_paths.py").is_file():
            sys.path.insert(0, str(_parent))
            break
    from rscdetr_paths import ROOT, DATASETS
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "WORKSPACE",
    "ROOT",
    "DATASETS",
    "VEDAI",
    "M3FD",
    "RTDOD",
    "DVTOD_COMPARE",
    "CFT",
    "LCAFNET",
    "MSOD",
    "PAPER",
    "OUT_QUAL",
    "describe",
]


def _env_path(name: str, default: Path | str) -> Path:
    """Resolve *name* from the environment, falling back to *default*."""
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else Path(default).expanduser()


# --------------------------------------------------------------------------
# workspace and main repository
# --------------------------------------------------------------------------
WORKSPACE = _env_path("RSCDETR_WORKSPACE", Path.home())
ROOT = _env_path("RSCDETR_ROOT", WORKSPACE / "rsc-detr")

# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
DATASETS = _env_path("RSCDETR_DATASETS", WORKSPACE / "vedai_data" / "datasets")
VEDAI = DATASETS / "VEDAI"
M3FD = DATASETS / "M3FD"

# DVTOD / RTDOD live inside the main repository rather than the dataset root
RTDOD = ROOT / "datasets" / "RTDOD_HBB_3class"
DVTOD_COMPARE = ROOT / "compare" / "DVTOD_compare_20260910"

# --------------------------------------------------------------------------
# comparison-method checkouts
# --------------------------------------------------------------------------
CFT = _env_path("RSCDETR_CFT", WORKSPACE / "CFT")
LCAFNET = _env_path("RSCDETR_LCAFNET", WORKSPACE / "LCAFNet")
MSOD = _env_path("RSCDETR_MSOD", WORKSPACE / "multispectral-object-detection")

# --------------------------------------------------------------------------
# paper sources and figure output
# --------------------------------------------------------------------------
PAPER = _env_path("RSCDETR_PAPER", ROOT / "RSC_DETR")
OUT_QUAL = _env_path("RSCDETR_OUT_QUAL", ROOT / "out_qual")

_ENV_VARS = {
    "WORKSPACE": "RSCDETR_WORKSPACE",
    "ROOT": "RSCDETR_ROOT",
    "DATASETS": "RSCDETR_DATASETS",
    "CFT": "RSCDETR_CFT",
    "LCAFNET": "RSCDETR_LCAFNET",
    "MSOD": "RSCDETR_MSOD",
    "PAPER": "RSCDETR_PAPER",
    "OUT_QUAL": "RSCDETR_OUT_QUAL",
}


def describe() -> str:
    """Return a human-readable dump of every resolved path."""
    rows = [
        ("WORKSPACE", WORKSPACE),
        ("ROOT", ROOT),
        ("DATASETS", DATASETS),
        ("  VEDAI", VEDAI),
        ("  M3FD", M3FD),
        ("RTDOD", RTDOD),
        ("DVTOD_COMPARE", DVTOD_COMPARE),
        ("CFT", CFT),
        ("LCAFNET", LCAFNET),
        ("MSOD", MSOD),
        ("PAPER", PAPER),
        ("OUT_QUAL", OUT_QUAL),
    ]
    lines = []
    for key, value in rows:
        mark = ""
        env_var = _ENV_VARS.get(key.strip())
        if env_var and os.environ.get(env_var, "").strip():
            mark = f"  (from ${env_var})"
        lines.append(f"{key:<14} {value}{mark}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
