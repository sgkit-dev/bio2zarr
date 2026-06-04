import numpy as np

INT_MISSING = -1
INT_FILL = -2
STR_MISSING = "."
STR_FILL = ""

FLOAT32_MISSING, FLOAT32_FILL = np.array([0x7F800001, 0x7F800002], dtype=np.int32).view(
    np.float32
)
FLOAT16_MISSING, FLOAT16_FILL = np.array([0x7C01, 0x7C02], dtype=np.int16).view(
    np.float16
)
FLOAT64_MISSING, FLOAT64_FILL = np.array(
    [0x7FF0000000000001, 0x7FF0000000000002], dtype=np.uint64
).view(np.float64)

# Maps a float dtype to its (missing, fill) sentinel pair.
FLOAT_MISSING_FILL = {
    np.dtype("float16"): (FLOAT16_MISSING, FLOAT16_FILL),
    np.dtype("float32"): (FLOAT32_MISSING, FLOAT32_FILL),
    np.dtype("float64"): (FLOAT64_MISSING, FLOAT64_FILL),
}


MIN_INT_VALUE = np.iinfo(np.int32).min + 2
VCF_INT_MISSING = np.iinfo(np.int32).min
VCF_INT_FILL = np.iinfo(np.int32).min + 1
