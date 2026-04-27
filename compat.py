import os
import sys


def default_project_root():
    """Return the local pixel_resshift repo root."""

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_repo_path(repo_root):
    root = os.path.abspath(os.path.expanduser(repo_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def ensure_pmf_torch_path(pmf_torch_root=None):
    """Make the local pMF_torch repo importable.

    pMF_torch is a source repo instead of an installed package, and its internal
    imports are written as ``from models ...`` / ``from utils ...``. We therefore
    add the repo root to ``sys.path`` before importing any of its modules.
    """

    if pmf_torch_root is None:
        pmf_torch_root = os.path.join(default_project_root(), "pMF_torch")
    return _ensure_repo_path(pmf_torch_root)


def ensure_resshift_path(resshift_root=None):
    """Make the local ResShift repo importable."""

    if resshift_root is None:
        resshift_root = os.path.join(default_project_root(), "ResShift")
    return _ensure_repo_path(resshift_root)
