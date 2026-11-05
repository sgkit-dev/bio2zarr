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
        scheme, _, key = uri.partition("://")
        parts = key.split("/", 1)
        if (
            scheme not in ("s3", "s3+http", "s3+https")
            or len(parts) != 2
            or not all(parts)
        ):
            raise ValueError("S3 paths must include a bucket and an object key/prefix")
        object.__setattr__(self, "uri", uri)

    def __str__(self):
        return self.uri

    @property
    def object_path(self):
        return "s3://" + self.uri.split("://", 1)[1]

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
        return self.fs.exists(self.object_path)

    def is_dir(self):
        return self.fs.isdir(self.object_path)

    def open(self, mode="r", **kwargs):
        return self.fs.open(self.object_path, mode, **kwargs)

    def mkdir(self, parents=False, exist_ok=False):
        # S3 directories are prefixes, not objects. Never create a bucket here.
        bucket = self.uri.split("://", 1)[1].split("/")[0]
        if not self.fs.exists(bucket):
            raise FileNotFoundError(f"S3 bucket does not exist: {bucket}")

    def unlink(self):
        self.fs.rm_file(self.object_path)

    @property
    def size(self):
        return self.fs.size(self.object_path)


def path(value):
    if isinstance(value, S3Path):
        return value
    if str(value).startswith(("s3://", "s3+http://", "s3+https://")):
        return S3Path(str(value))
    if "://" in str(value):
        raise ValueError(f"Unsupported storage protocol: {str(value).split('://')[0]}")
    return pathlib.Path(value)


def remove_tree(value):
    value = path(value)
    if isinstance(value, S3Path):
        # Match only descendants: a similarly named neighbouring prefix is not ours.
        objects = value.fs.find(value.object_path + "/")
        if objects:
            value.fs.rm(objects)
    else:
        shutil.rmtree(value)


def file_size(value):
    value = path(value)
    return value.size if isinstance(value, S3Path) else value.stat().st_size
