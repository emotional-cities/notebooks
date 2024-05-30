from datetime import timedelta
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib.gridspec import GridSpec
from pandas import DataFrame, concat
from pluma.schema import Dataset
from pluma.stream.georeference import Georeference
from pluma.stream.eeg import NedfReader
from pluma.export.maps import showmap
from pluma.io.path_helper import ensure_complexpath
from pluma.sync.plotting import plot_clockcalibration_diagnosis
import numpy as np
import cv2 as cv

def load_dataset(root, schema, reload=True, export_path=None):
    # Path to the dataset. Can be local or remote.
    dataset = Dataset(
        root=root,
        datasetlabel="FMUL_" + root.split("\\")[-1],
        georeference= Georeference(),
        schema=schema)  # Create a Dataset object that will contain the ingested data.
    dataset.populate_streams(autoload=False)  # Add the "schema" that we want to load to our Dataset. If we want to load the whole dataset automatically, set autoload to True.

    if reload:
        # We will just load every single stream at the same time. This might take a while if loading from AWS
        # Some warnings will be printed if some sensors were not acquired during the experiment. These are normal and can be usually ignored.
        dataset.reload_streams(force_load=True)
        sync_lookup = dataset.calibrate_ubx_to_harp()
        dataset.add_ubx_georeference()
        dataset.reference_harp_to_ubx_time()
        dataset.sync_lookup = sync_lookup


        if export_path is not None:
            # We can export the dataset as a .pickle file.
            # In order to not having to run this routine multiple times, the output of the
            # ingestion can be saved as a pickle file to be loaded later. E.g.:
            dataset.export_dataset(filename=f"{export_path}\dataset.pickle")

    else:
        dataset = Dataset.import_dataset(f"{root}\dataset.pickle")
        # ... and reimport it at a later point.

    return dataset

def eeg_segments(dataset):
    eeg_time = dataset.streams.EEG.data.np_time
    eeg_markers = dataset.streams.EEG.server_lsl_marker

    trial_markers = eeg_markers[eeg_markers.MarkerIdx > 35200]
    trial_id = trial_markers.MarkerIdx - 35200

    trial_ids = concat([trial_markers.Seconds, trial_id], axis=1)
    row0 = DataFrame([(eeg_time[0], 0)], columns=trial_ids.columns)
    trials = concat([row0, trial_ids], axis=0, ignore_index=True)
    return trials.set_index('Seconds')

def periodic_segments(dataset, slice_dt='5min'):
    df = dataset.georeference.elevation.resample(slice_dt).count()
    segments = DataFrame((marker for marker in df.index), columns=['Seconds'])
    segments['MarkerIdx'] = segments.index
    return segments.set_index('Seconds')

def plot_summary(dataset):
    sync_lookup = dataset.sync_lookup
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, hspace=0.5, figure=fig)
    idx_ax = fig.add_subplot(gs[0, 0])
    scatter_ax = fig.add_subplot(gs[0, 1])
    map_ax = fig.add_subplot(gs[1:, :])
    plot_clockcalibration_diagnosis(
        sync_lookup.ubx_ts,
        sync_lookup.harp_ts,
        sync_lookup.align_lookup,
        axes=(idx_ax, scatter_ax))
    dataset.showmap(
        fig=fig,
        ax=map_ax)
    plt.show()

def plot_stream(stream):
    stream.plot()
    plt.show()

def plot_path(dataset, sampling_dt=timedelta(seconds=2), colorscale_override=None, **kwargs):
    fig = dataset.showmap(
        colorscale_override=dataset.georeference.spacetime.index
            if colorscale_override is None
            else colorscale_override,
        cmap = "jet",
        **kwargs)
    fig.get_axes()[0].set_title(dataset.datasetlabel)
    plt.show()

def plot_geospatial(data, sampling_dt=timedelta(seconds=2), **kwargs):
    fig = showmap(data, **kwargs)
    fig.get_axes()[0].set_title(kwargs.get('title', 'Data'))
    plt.show()

def plot_traces(traces, segments=None, figsize = (10,4)):
    ## Plot it in time for comparison
    fig, axs = plt.subplots(len(traces),1, sharex=True)
    fig.set_size_inches(figsize)

    if segments is None:
        segments = [(None, [0.1, 0.1, 0.1])]

    for si in range(len(segments)):
        marker, color = segments[si]
        end, _ = segments[si + 1] if si < len(segments)-1 else (None, None)
        for i, (label, data) in enumerate(traces.items()):
            segment = slice(marker, end)
            if isinstance(data, NedfReader):
                eeg_time = DataFrame(np.arange(len(data.np_time)), index=data.np_time)
                eeg_segment = eeg_time.loc[segment].values.ravel()
                axs[i].set_prop_cycle(None)
                axs[i].plot(data.np_time[eeg_segment],
                            data.np_eeg[eeg_segment],
                            c = color,
                            lw = 0.5,
                            alpha=0.3)
            else:
                axs[i].plot(data.loc[segment], c = color, lw = 0.5)
            axs[i].set_ylabel(label)
    fig.supxlabel('Time')
    fig.align_ylabels()
    plt.show()

def load_pupilvideo(dataset):
    root_path = ensure_complexpath(dataset.rootfolder)
    root_path.join('pupil_video.avi')
    return cv.VideoCapture(root_path.path)

def load_pupilgaze(dataset):
    root_path = ensure_complexpath(dataset.rootfolder)
    root_path.join('PupilLabs')
    root_path.join('Gaze_Frame2.bin')

    with root_path.open('rb') as stream:
        data = np.frombuffer(stream.read(), dtype=np.float32).reshape((-1, 2))
    frame_counter = dataset.streams.PupilLabs.Counter.Gaze.data
    data = DataFrame(data, columns=['x', 'y'])
    data.index = frame_counter.index
    return data

def reindex_pupilgaze(dataset, gaze):
    # drop duplicate timestamps
    gaze = gaze.groupby(gaze.index).first()
    decoded_frames = dataset.streams.PupilLabs.Counter.DecodedFrames.data
    decoded_frames = decoded_frames[decoded_frames.Value > 0] # drop null frames
    gaze = gaze.reindex(decoded_frames.index, method='pad')
    gaze['frame'] = np.arange(len(gaze))
    location = dataset.georeference.spacetime.reindex(gaze.index, method='nearest')
    return gaze.join(location)

def plot_example_traces(dataset, traces, **kwargs):
    segment_colors = np.array(['red', 'green', 'magenta', 'black', 'blue'])
    marker_segments = eeg_segments(dataset)

    path_segments = marker_segments.reindex(
        dataset.georeference.elevation.index,
        method='pad').MarkerIdx.fillna(0).astype('int')
    plot_path(dataset, colorscale_override=segment_colors[path_segments], **kwargs)

    segments = [(x, segment_colors[idx % len(segment_colors)])
                for (x, idx) in marker_segments.reset_index().values]
    plot_traces(traces, segments, figsize=(4.5,8))
