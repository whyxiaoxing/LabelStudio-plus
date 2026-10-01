"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license."""

import configparser
import logging
import os
import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

from .settings.base import EXPORT_DIR

logger = logging.getLogger(__name__)

CONF_FILENAME = 'LabelStudio.conf'
CONF_SECTION = 'Server'
CONF_PATH = pathlib.Path(__file__).resolve().parent.parent / CONF_FILENAME

# sub-commands are still given on the command line, every other option lives in LabelStudio.conf
COMMANDS = (
    'version',
    'user',
    'init',
    'start',
    'reset_password',
    'shell',
    'calculate_stats_all_orgs',
    'export',
    'annotations_fill_updated_by',
)

TRUE_VALUES = ('1', 'true', 'yes', 'on')
FALSE_VALUES = ('0', 'false', 'no', 'off')

BOOL_OPTIONS = (
    'debug',
    'no_browser',
    'enable_legacy_api_token',
    'quiet_mode',
    'init',
    'version',
    'from_scratch',
)
INT_OPTIONS = ('port',)
LIST_OPTIONS = ('ml_backends',)
PATH_OPTIONS = ('database', 'data_dir', 'label_config', 'cert_file', 'key_file', 'export_path')

DEFAULT_EXPORT_SERIALIZER_CONTEXT = '{"annotations__completed_by": {"only_id": null}, "interpolate_key_frames": true}'

DEFAULTS: Dict[str, Any] = {
    'host': '',
    'internal_host': '0.0.0.0',  # nosec
    'port': None,
    'debug': False,
    'no_browser': False,
    'log_level': 'WARNING',
    'database': None,
    'data_dir': None,
    'label_config': None,
    'cert_file': None,
    'key_file': None,
    'username': '',
    'password': '',
    'user_token': '',
    'enable_legacy_api_token': False,
    'quiet_mode': False,
    'project_name': '',
    'project_desc': None,
    'sampling': 'sequential',
    'ml_backends': None,
    'init': False,
    'version': False,
    'from_scratch': False,
    'project_id': None,
    'export_format': None,
    'export_path': EXPORT_DIR,
    'export_serializer_context': DEFAULT_EXPORT_SERIALIZER_CONTEXT,
}


def _warn(option: str, value: str, expected: str) -> None:
    logger.warning(f'Option {option} = {value} in {CONF_PATH.name} is not {expected}, the default is used')


def _parse_bool(option: str, value: str) -> Optional[bool]:
    """Parse a boolean option, ``None`` means the value is not recognizable and must be ignored."""
    raw = value.lower()
    if raw in TRUE_VALUES:
        return True
    if raw in FALSE_VALUES:
        return False
    _warn(option, value, 'a boolean')
    return None


def _parse_int(option: str, value: str) -> Optional[int]:
    """Parse an integer option, ``None`` means the value is not recognizable and must be ignored."""
    try:
        return int(value)
    except ValueError:
        _warn(option, value, 'an integer')
        return None


def _parse_list(option: str, value: str) -> List[str]:
    """Parse a comma separated option into a list, e.g. ``a,b`` => ``['a', 'b']``."""
    return [item.strip() for item in value.split(',') if item.strip()]


def _parse_path(option: str, value: str) -> str:
    """Expand a path option into an absolute path."""
    return os.path.abspath(os.path.expanduser(value))


_PARSERS: Dict[str, Callable[[str, str], Any]] = {
    **{option: _parse_bool for option in BOOL_OPTIONS},
    **{option: _parse_int for option in INT_OPTIONS},
    **{option: _parse_list for option in LIST_OPTIONS},
    **{option: _parse_path for option in PATH_OPTIONS},
}


def _read_conf(conf_path: pathlib.Path) -> Dict[str, str]:
    """Read the ``[Server]`` section of LabelStudio.conf, every value is kept as a raw string.

    Args:
        conf_path: Path to the config file.

    Returns:
        Mapping of the lowercased option names to their raw values, empty when the file or the
        section is missing.
    """
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with open(conf_path, encoding='utf-8') as conf_file:
            parser.read_file(conf_file)
    except OSError as e:
        logger.warning(f'Can not read {conf_path}: {e}, built-in defaults will be used')
        return {}
    except configparser.Error as e:
        logger.warning(f'Can not parse {conf_path}: {e}, built-in defaults will be used')
        return {}

    if not parser.has_section(CONF_SECTION):
        logger.warning(f'Section [{CONF_SECTION}] is missing in {conf_path}, built-in defaults will be used')
        return {}

    return {option.strip().lower(): value for option, value in parser[CONF_SECTION].items()}


def _get_command(argv: Sequence[str], options: Dict[str, str]) -> Optional[str]:
    """Pick the sub-command, it is the only option still given on the command line."""
    for arg in argv:
        if arg in COMMANDS:
            return arg
    command = options.get('command', '').strip().lower()
    return command if command in COMMANDS else None


def load_server_args(
    argv: Optional[Sequence[str]] = None, conf_path: Optional[pathlib.Path] = None
) -> SimpleNamespace:
    """Build the server options from LabelStudio.conf instead of the command line.

    The sub-command is still taken from the command line, e.g. ``label-studio start``, while every
    other option is read from the ``[Server]`` section of ``LabelStudio.conf`` sitting next to
    ``server.py``. Empty values and unknown options fall back to the built-in defaults.

    Args:
        argv: Command line arguments without the program name, ``sys.argv[1:]`` by default.
        conf_path: Path to the config file, ``LabelStudio.conf`` next to ``server.py`` by default.

    Returns:
        Namespace with the same attributes the command line parser used to return.
    """
    if argv is None:
        argv = sys.argv[1:]
    if conf_path is None:
        conf_path = CONF_PATH
    conf_path = pathlib.Path(conf_path)

    raw_options = _read_conf(conf_path)
    options: Dict[str, Any] = dict(DEFAULTS)

    for name, raw_value in raw_options.items():
        if name == 'command':
            continue
        if name not in DEFAULTS:
            logger.warning(f'Unknown option {name} in {conf_path}, it is ignored')
            continue
        value = raw_value.strip()
        if not value:
            continue
        parse = _PARSERS.get(name)
        parsed = parse(name, value) if parse else value
        if parsed is not None:
            options[name] = parsed

    options['command'] = _get_command(argv, raw_options)
    return SimpleNamespace(**options)
