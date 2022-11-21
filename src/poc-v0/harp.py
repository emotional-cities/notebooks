import datetime
import numpy as np
import pandas as pd

HARP_ORIGIN = datetime.datetime(1904, 1, 1)

payloadtypes = {
    1 : np.dtype(np.uint8),
    2 : np.dtype(np.uint16),
    4 : np.dtype(np.uint32),
    8 : np.dtype(np.uint64),
    129 : np.dtype(np.int8),
    130 : np.dtype(np.int16),
    132 : np.dtype(np.int32),
    136 : np.dtype(np.int64),
    68 : np.dtype(np.float32)
}

def read(file, names=None, origin=None):
    '''
    Read single-register Harp data from the specified file.
    
    :param str file: The path to a Harp binary file containing data from a single device register.
    :param str or array-like names: The optional column labels to use for the data values.
    :param datetime origin: The optional time used as timestamp reference.
    :return: A pandas data frame containing harp event data, sorted by time.
    '''
    if origin is None:
        origin = HARP_ORIGIN

    if isinstance(file, str):
        data = np.fromfile(file, dtype=np.uint8)
    else:
        try:
            data = np.frombuffer(file.read(), dtype=np.uint8)
        finally:
            file.close()
    stride = data[1] + 2
    length = len(data) // stride
    payloadsize = stride - 12
    payloadtype = payloadtypes[data[4] & ~0x10]
    elementsize = payloadtype.itemsize
    payloadshape = (length, payloadsize // elementsize)
    iseconds = np.ndarray(length, dtype=np.uint32, buffer=data, offset=5, strides=stride)
    micros = np.ndarray(length, dtype=np.uint16, buffer=data, offset=9, strides=stride)
    seconds = micros * 32e-6 + iseconds
    payload = np.ndarray(
        payloadshape,
        dtype=payloadtype,
        buffer=data, offset=11,
        strides=(stride, elementsize))
    time = origin + pd.to_timedelta(seconds, 's')
    time.name = 'Timestamp'
    return pd.DataFrame(payload, index=time, columns=names)