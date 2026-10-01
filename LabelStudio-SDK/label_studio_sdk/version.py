from importlib import metadata

try:
    __version__ = metadata.version("label-studio-sdk")
except metadata.PackageNotFoundError:
    # The SDK ships inside the label-studio distribution in this repository, so
    # there is no `label-studio-sdk` dist-info to read the version from.
    __version__ = "2.1.2"
