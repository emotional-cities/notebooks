import os
from IPython.display import clear_output, display
from ipyfilechooser import FileChooser
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
from numpy import ndarray
import tilemapbase as tmb
from pathlib import Path

from pluma.export.maps import showmap, exploremap
from pluma.export.ogcapi.features import export_geoframe_to_geojson
from pluma.preprocessing.resampling import resample_temporospatial
from pluma.preprocessing.ecg import heartrate_from_ecg
from pluma.export.ogcapi.records import DatasetRecord, RecordProperties, Contact
from pluma.stream.georeference import Georeference
from pluma.stream.harp import HarpStream
from pluma.stream.ubx import UbxStream
from pluma.io.ubx import _UBX_MSGIDS, _UBX_CLASSES
from pluma.schema import Dataset
from schema import custom_schema
from helpers import *

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

## Figure export parameters
new_rc_params = {
    'text.usetex': False,
    'svg.fonttype': 'none'
}
import matplotlib as mpl
mpl.rcParams.update(new_rc_params)

## Ensure tilemapbase cache is initialized
tmb.init(create=True)

def create_datapicker(path=None, show_summary=True, ubx=True, create_geodata=True, schema=custom_schema):
    def dataset_changed(chooser):
        clear_output(wait=False)
        display(chooser)
        print(f"Loading dataset: {Path(chooser.selected_path).name}..." )
        dataset = load_dataset(chooser.selected_path, ubx=ubx, schema=schema)
        print(f"Dataset: {dataset} loaded successfully, and {'not' if not dataset.has_calibration else 'sucessfully'} calibrated.")
        chooser.dataset = dataset
        if show_summary:
            plot_summary(dataset)
        if create_geodata:
            chooser.geodata = dataset.to_geoframe()

    if path is None:
        config_path = Path(__file__).parent.absolute() / 'ingestion.config'
        if config_path.exists():
            with open(config_path) as config:
                path = config.read()
        else:
            path =  os.getcwd()

    datapicker = FileChooser(
        path=path,
        title='<b>Select the Dataset folder</b>',
        show_hidden=False,
        select_default=True,
        show_only_dirs=True
    )
    datapicker.register_callback(dataset_changed)
    return datapicker
