import numpy as np
import zarr

from bio2zarr import constants


def zarr_v3() -> bool:
    return zarr.__version__ >= "3"


if zarr_v3():
    # Use zarr format v2 even when running with zarr-python v3
    ZARR_FORMAT_KWARGS = dict(zarr_format=2)
else:
    ZARR_FORMAT_KWARGS = dict()


# See discussion in https://github.com/zarr-developers/zarr-python/issues/2529
def first_dim_iter(z):
    for chunk in range(z.cdata_shape[0]):
        yield from z.blocks[chunk]


def sanitise_value_bool(buff, j, value):
    x = True
    if value is None:
        x = False
    buff[j] = x


def sanitise_value_float_scalar(buff, j, value):
    x = value
    if value is None:
        x = [constants.FLOAT32_MISSING]
    buff[j] = x[0]


def sanitise_value_int_scalar(buff, j, value):
    x = value
    if value is None:
        # print("MISSING", INT_MISSING, INT_FILL)
        x = [constants.INT_MISSING]
    else:
        x = sanitise_int_array(value, ndmin=1, dtype=np.int32)
    buff[j] = x[0]


def sanitise_value_string_scalar(buff, j, value):
    if value is None:
        buff[j] = "."
    else:
        buff[j] = value[0]


def sanitise_value_string_1d(buff, j, value):
    if value is None:
        buff[j] = "."
    else:
        # value = np.array(value, ndmin=1, dtype=buff.dtype, copy=False)
        # FIXME failure isn't coming from here, it seems to be from an
        # incorrectly detected dimension in the zarr array
        # The dimesions look all wrong, and the dtype should be Object
        # not str
        value = drop_empty_second_dim(value)
        buff[j] = ""
        buff[j, : value.shape[0]] = value


def sanitise_value_string_2d(buff, j, value):
    if value is None:
        buff[j] = "."
    else:
        # print(buff.shape, value.dtype, value)
        # assert value.ndim == 2
        buff[j] = ""
        if value.ndim == 2:
            buff[j, :, : value.shape[1]] = value
        else:
            # TODO check if this is still necessary
            for k, val in enumerate(value):
                buff[j, k, : len(val)] = val


def drop_empty_second_dim(value):
    assert len(value.shape) == 1 or value.shape[1] == 1
    if len(value.shape) == 2 and value.shape[1] == 1:
        value = value[..., 0]
    return value


def sanitise_value_float_1d(buff, j, value):
    if value is None:
        buff[j] = constants.FLOAT32_MISSING
    else:
        value = np.array(value, ndmin=1, dtype=buff.dtype, copy=True)
        # numpy will map None values to Nan, but we need a
        # specific NaN
        value[np.isnan(value)] = constants.FLOAT32_MISSING
        value = drop_empty_second_dim(value)
        buff[j] = constants.FLOAT32_FILL
        buff[j, : value.shape[0]] = value


def sanitise_value_float_2d(buff, j, value):
    if value is None:
        buff[j] = constants.FLOAT32_MISSING
    else:
        # print("value = ", value)
        value = np.array(value, ndmin=2, dtype=buff.dtype, copy=True)
        buff[j] = constants.FLOAT32_FILL
        buff[j, :, : value.shape[1]] = value


def sanitise_int_array(value, ndmin, dtype):
    if isinstance(value, tuple):
        value = [
            constants.VCF_INT_MISSING if x is None else x for x in value
        ]  # NEEDS TEST
    value = np.array(value, ndmin=ndmin, copy=True)
    value[value == constants.VCF_INT_MISSING] = -1
    value[value == constants.VCF_INT_FILL] = -2
    # TODO watch out for clipping here!
    return value.astype(dtype)


def sanitise_value_int_1d(buff, j, value):
    if value is None:
        buff[j] = -1
    else:
        value = sanitise_int_array(value, 1, buff.dtype)
        value = drop_empty_second_dim(value)
        buff[j] = -2
        buff[j, : value.shape[0]] = value


def sanitise_value_int_2d(buff, j, value):
    if value is None:
        buff[j] = -1
    else:
        value = sanitise_int_array(value, 2, buff.dtype)
        buff[j] = -2
        buff[j, :, : value.shape[1]] = value
