from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("mypkg-name")
except PackageNotFoundError:  # during local dev without install
    __version__ = "0.0.0"