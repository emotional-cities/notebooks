from datetime import timedelta
from matplotlib import pyplot as plt
from pandas import DataFrame, concat
from numpy import arange, ndarray
from pluma.schema import Dataset
from pluma.stream.georeference import Georeference
from pluma.stream.eeg import NedfReader
from pluma.preprocessing.resampling import resample_temporospatial
from pluma.export.maps import showmap

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
        dataset.add_georeference_and_calibrate()

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
    fig.show()

def plot_geospatial(data, sampling_dt=timedelta(seconds=2), **kwargs):
    fig = showmap(data, **kwargs)
    fig.get_axes()[0].set_title(kwargs.get('title', 'Data'))
    fig.show()

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
                eeg_time = DataFrame(arange(len(data.np_time)), index=data.np_time)
                eeg_segment = eeg_time.loc[segment].values.ravel()
                axs[i].set_prop_cycle(None)
                axs[i].plot(data.np_time[eeg_segment],
                            data.np_eeg[eeg_segment],
                            c = color,
                            lw = 0.5)
            else:
                axs[i].plot(data.loc[segment], c = color, lw = 0.5)
            axs[i].set_ylabel(label)
    fig.supxlabel('Time')
    fig.align_ylabels()
    fig.show()
