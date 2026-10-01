"""
Utility helpers for LocalFiles storage operations.
"""

import os
import posixpath
from pathlib import Path
from typing import Callable, Iterable, Optional

from django.conf import settings
from django.core.exceptions import SuspiciousFileOperation
from django.utils._os import safe_join

AUTO_ROOT_CANDIDATES: tuple[str, ...] = ('mydata', 'label-studio-data')


def autodetect_local_files_root(
    base_dir: Optional[str] = None, candidates: Iterable[str] = AUTO_ROOT_CANDIDATES
) -> Optional[str]:
    """Return the first existing candidate directory relative to ``base_dir``."""
    search_root = Path(base_dir or os.getcwd())
    for candidate_name in candidates:
        candidate = (search_root / candidate_name).expanduser()
        if candidate.is_dir():
            return str(candidate.resolve())

    return None


def normalize_storage_path(raw_path: str | None) -> str | None:
    """Return a canonical representation for LocalFiles paths.

    We need consistent paths because permission checks compare the requested
    directory with storage.path prefixes. Users often enter trailing slashes or
    Windows separators; normalizing here prevents mismatches.
    """
    if raw_path is None:
        return None

    trimmed = raw_path.strip()
    if trimmed == '':
        return ''

    collapsed = trimmed.replace('\\', os.sep)
    normalized = os.path.normpath(collapsed)
    return normalized


def build_local_files_path(relative_path: str) -> str:
    """Map a ``/data/local-files/?d=`` value to a path under LOCAL_FILES_DOCUMENT_ROOT.

    Raises SuspiciousFileOperation when the value escapes the document root.
    """
    # Normalize the incoming relative path so we don't depend on trailing slashes.
    # Backslashes are folded to "/" first: Local Storage is configured with either
    # separator, and the SDK downloader applies the same rule in io.py, so the file
    # a later download reads is the file this resolver approved.
    # posixpath.normpath, not Path: collapsing ".." is what makes the escape checks
    # below work (test_localfiles_export_resolver relies on it), and Path keeps it.
    normalized = posixpath.normpath(str(relative_path).replace('\\', '/')).lstrip('/')
    return safe_join(settings.LOCAL_FILES_DOCUMENT_ROOT, normalized)


def is_within_storage(directory: str, storage_path: str) -> bool:
    # Structural, not a string prefix: storage /data/ds must not match /data/ds-private
    # the way SQL's startswith would. Path.is_relative_to compares whole components and,
    # on Windows, ignores case the way the filesystem does - startswith() did neither.
    if not directory or not storage_path:
        return False
    return Path(directory).is_relative_to(storage_path)


def project_local_files_resolver(project) -> Callable[[str], Optional[str]]:
    """Build the SDK converter's Local Storage resolver for exports of ``project``.

    Uses the path rules of the /data/local-files/ view: nothing is readable while local files serving
    is off, otherwise only files under one of this project's Local Files storages.
    """
    # Imported lazily: this module is loaded by settings before the app registry is ready
    from io_storages.localfiles.models import LocalFilesImportStorage

    storage_paths = []
    if settings.LOCAL_FILES_SERVING_ENABLED:
        storage_paths = list(LocalFilesImportStorage.objects.filter(project=project).values_list('path', flat=True))

    def resolve(relative_path: str) -> Optional[str]:
        try:
            full_path = build_local_files_path(relative_path)
        except SuspiciousFileOperation:
            return None
        directory = str(Path(full_path).parent)
        if any(is_within_storage(directory, path) for path in storage_paths) and os.path.isfile(full_path):
            return full_path
        return None

    return resolve
