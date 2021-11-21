import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

def _butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = _butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def ecgpeaks(ecg, **kwargs):
    peaks, peakattributes = find_peaks(ecg, **kwargs)
    peaks = ecg.index[peaks]
    return peaks, peakattributes

def heartrate(ecg, window=None, **kwargs):
    beats, _ = ecgpeaks(ecg, **kwargs)
    ibi = beats.to_series().diff().dt.total_seconds()
    return 60 / ibi.rolling(window=pd.to_timedelta(60, 's')).mean()