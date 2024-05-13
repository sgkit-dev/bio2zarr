import numpy as np

INT_MISSING = -1
INT_FILL = -2
STR_MISSING = "."
STR_FILL = ""

FLOAT32_MISSING, FLOAT32_FILL = np.array([0x7F800001, 0x7F800002], dtype=np.int32).view(
    np.float32
)
FLOAT32_MISSING_AS_INT32, FLOAT32_FILL_AS_INT32 = np.array(
    [0x7F800001, 0x7F800002], dtype=np.int32
)


MIN_INT_VALUE = np.iinfo(np.int32).min + 2
VCF_INT_MISSING = np.iinfo(np.int32).min
VCF_INT_FILL = np.iinfo(np.int32).min + 1
