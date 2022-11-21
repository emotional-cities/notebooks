import wave
import numpy as np
import pandas as pd

def wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.frombuffer(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.frombuffer(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result

def readaudio(name):
    audio = wave.open(name)
    nchannels = audio.getnchannels()
    sampwidth = audio.getsampwidth()
    nsamples = audio.getnframes()
    rawdata = audio.readframes(nsamples)
    audio.close()
    return wav2array(nchannels,sampwidth,rawdata)

def readsamples(audio, start=0, nsamples=None):
    nchannels = audio.getnchannels()
    sampwidth = audio.getsampwidth()
    if nsamples is None:
        nsamples = audio.getnframes() - start
    audio.setpos(start)
    rawdata = audio.readframes(nsamples)
    return wav2array(nchannels,sampwidth,rawdata)

def time_slice(audio, audiotime, start_time, end_time, samples_per_buffer=441):
    index = pd.DatetimeIndex(audiotime)
    buffer = index.slice_indexer(start_time, end_time)
    start = buffer.start * samples_per_buffer
    nsamples = buffer.stop * samples_per_buffer - start
    data = pd.DataFrame(readsamples(audio, start, nsamples))
    data.index = pd.date_range(start=index[buffer.start], end=index[buffer.stop], periods=nsamples)
    data.columns = ['Left', 'Right']
    return data

def between_time(audio, audiotime, start_time, end_time, samples_per_buffer=441):
    index = pd.DatetimeIndex(audiotime)
    buffers = index.indexer_between_time(start_time, end_time)
    start = buffers[0] * samples_per_buffer
    nsamples = buffers[-1] * samples_per_buffer - start
    data = pd.DataFrame(readsamples(audio, start, nsamples))
    data.index = pd.date_range(start=index[buffers[0]], end=index[buffers[-1]], periods=nsamples)
    data.columns = ['Left', 'Right']
    return data