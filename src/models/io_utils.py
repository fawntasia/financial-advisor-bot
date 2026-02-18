"""IO and artifact helpers for model scripts."""

from __future__ import annotations

import os
import sys
from importlib import metadata
from typing import Dict, Iterable


def ensure_parent_dir(path: str) -> None:
    """Create a file's parent directory if it exists in the path."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_library_versions(packages: Iterable[str] | None = None) -> Dict[str, str]:
    """Return installed versions for a set of relevant runtime packages."""
    default_packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "tensorflow",
        "joblib",
    )
    package_names = tuple(packages) if packages is not None else default_packages

    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for package in package_names:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions
