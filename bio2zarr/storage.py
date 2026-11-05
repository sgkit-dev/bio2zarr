"""Local paths and S3 object prefixes used by VCF and intermediate files."""

import os
import pathlib
import shutil
from dataclasses import dataclass

from . import core


@dataclass(frozen=True)
class S3Path:
    uri: str

    def __post_init__(self):
        uri = self.uri.rstrip("/")
        parts = uri.removeprefix("s3://").split("/", 1)
        if not uri.startswith("s3://") or len(parts) != 2 or not all(parts):
            raise ValueError("S3 paths must include a bucket and an object key/prefix")
        object.__setattr__(self, "uri", uri)

    def __str__(self):
        return self.uri

    def __truediv__(self, name):
        return S3Path(f"{self.uri}/{name}")

    @property
    def suffix(self):
        return pathlib.PurePosixPath(self.uri).suffix

    def with_suffix(self, suffix):
        parent, name = self.uri.rsplit("/", 1)
        return S3Path(f"{parent}/{pathlib.PurePosixPath(name).with_suffix(suffix)}")

    @property
    @core.requires_optional_dependency("s3fs", "s3")
    def fs(self):
        import s3fs  # noqa: PLC0415

        return s3fs.S3FileSystem(
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3")
            or os.environ.get("AWS_ENDPOINT_URL"),
            use_listings_cache=False,
            config_kwargs={"signature_version": "s3v4"},
        )

    def exists(self):
        return self.fs.exists(self.uri)

    def is_dir(self):
        return self.fs.isdir(self.uri)

    def open(self, mode="r", **kwargs):
        return self.fs.open(self.uri, mode, **kwargs)

    def mkdir(self, parents=False, exist_ok=False):
        # S3 directories are prefixes, not objects. Never create a bucket here.
        bucket = self.uri.removeprefix("s3://").split("/")[0]
        if not self.fs.exists(bucket):
            raise FileNotFoundError(f"S3 bucket does not exist: {bucket}")

    def unlink(self):
        self.fs.rm_file(self.uri)

    @property
    def size(self):
        return self.fs.size(self.uri)

    def url(self):
        # URLs are refreshed for every opened partition, never written to metadata.
        return self.fs.url(self.uri, expires=86400)


def path(value):
    if isinstance(value, S3Path):
        return value
    if str(value).startswith("s3://"):
        return S3Path(str(value))
    if "://" in str(value):
        raise ValueError(f"Unsupported storage protocol: {str(value).split('://')[0]}")
    return pathlib.Path(value)


def remove_tree(value):
    value = path(value)
    if isinstance(value, S3Path):
        # Match only descendants: a similarly named neighbouring prefix is not ours.
        objects = value.fs.find(str(value) + "/")
        if objects:
            value.fs.rm(objects)
    else:
        shutil.rmtree(value)


def file_size(value):
    value = path(value)
    return value.size if isinstance(value, S3Path) else value.stat().st_size
