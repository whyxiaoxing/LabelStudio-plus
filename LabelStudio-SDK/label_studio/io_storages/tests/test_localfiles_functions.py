import os
import tempfile

import pytest
from django.core.exceptions import SuspiciousFileOperation
from io_storages.localfiles.functions import (
    autodetect_local_files_root,
    build_local_files_path,
    is_within_storage,
    normalize_storage_path,
)


def test_autodetect_local_files_root_returns_first_candidate(tmp_path, monkeypatch):
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    target_dir = project_dir / 'mydata'
    target_dir.mkdir()
    (project_dir / 'label-studio-data').mkdir()

    monkeypatch.chdir(project_dir)

    detected = autodetect_local_files_root()

    assert detected == str(target_dir.resolve())


def test_autodetect_local_files_root_returns_none_when_missing(tmp_path):
    base_dir = tmp_path / 'project'
    base_dir.mkdir()

    detected = autodetect_local_files_root(base_dir=str(base_dir))

    assert detected is None


_TMP_DIR = tempfile.gettempdir()
_DATASET_DIR = os.path.join(_TMP_DIR, 'dataset')


@pytest.mark.parametrize(
    'raw, expected',
    [
        (None, None),
        ('', ''),
        (_DATASET_DIR, os.path.normpath(_DATASET_DIR)),
        (_DATASET_DIR + os.sep, os.path.normpath(_DATASET_DIR)),
        (f'  {_DATASET_DIR}{os.sep}  ', os.path.normpath(_DATASET_DIR)),
        (os.path.join(_TMP_DIR, 'dataset', ''), os.path.join(_TMP_DIR, 'dataset')),
        (_DATASET_DIR.replace(os.sep, '\\') + '\\', os.path.join(_TMP_DIR, 'dataset')),
    ],
)
def test_normalize_storage_path_basic_cases(raw, expected):
    assert normalize_storage_path(raw) == expected


def test_normalize_storage_path_windows_drive():
    raw = 'C:\\data\\set\\'
    expected = 'C:/data/set' if os.name != 'nt' else 'C:\\data\\set'
    assert normalize_storage_path(raw) == expected


def test_build_local_files_path_accepts_either_separator(settings, tmp_path):
    """?d= values are typed by hand, so Windows separators reach this on Linux too."""
    settings.LOCAL_FILES_DOCUMENT_ROOT = str(tmp_path)

    assert build_local_files_path('dataset/cat.jpg') == str(tmp_path / 'dataset' / 'cat.jpg')
    assert build_local_files_path('dataset\\cat.jpg') == str(tmp_path / 'dataset' / 'cat.jpg')
    assert build_local_files_path('/dataset/cat.jpg') == str(tmp_path / 'dataset' / 'cat.jpg')


def test_build_local_files_path_still_rejects_escapes_after_normalizing(settings, tmp_path):
    """Folding backslashes must not open a way out of the document root."""
    settings.LOCAL_FILES_DOCUMENT_ROOT = str(tmp_path)

    with pytest.raises(SuspiciousFileOperation):
        build_local_files_path('..\\..\\etc\\passwd')
    with pytest.raises(SuspiciousFileOperation):
        build_local_files_path('dataset/../../etc/passwd')


def test_is_within_storage_compares_components_not_text():
    """A sibling whose name merely extends the storage path is not inside it."""
    assert is_within_storage(_DATASET_DIR, _DATASET_DIR)
    assert is_within_storage(os.path.join(_DATASET_DIR, 'sub'), _DATASET_DIR)
    assert is_within_storage(os.path.join(_DATASET_DIR, 'sub'), _DATASET_DIR + os.sep)
    assert not is_within_storage(_DATASET_DIR + '-private', _DATASET_DIR)
    assert not is_within_storage(os.path.join(_TMP_DIR, 'datasetXsub'), _DATASET_DIR)


def test_is_within_storage_is_false_for_blank_paths():
    """Storage paths can be saved empty; nothing lives inside ""."""
    assert not is_within_storage(_DATASET_DIR, '')
    assert not is_within_storage('', _DATASET_DIR)
