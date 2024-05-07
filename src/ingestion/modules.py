import os
from IPython.display import clear_output
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
from pluma.stream.ubx import UbxStream, _UBX_MSGIDS, _UBX_CLASSES
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

def create_datapicker():
    def dataset_changed(chooser):
        clear_output(wait=False)
        print(f"Loading dataset: {Path(chooser.selected_path).name}..." )
        dataset = load_dataset(chooser.selected_path, schema=custom_schema)
        print(f"Dataset: {dataset} loaded successfully, and {'not' if not dataset.has_calibration else 'sucessfully'} calibrated." )
        plot_summary(dataset)
        chooser.dataset = dataset
        chooser.geodata = dataset.to_geoframe()

    datapicker = FileChooser(
        os.getcwd(),
        title='<b>Select the Dataset folder</b>',
        show_hidden=False,
        select_default=True,
        show_only_dirs=True
    )
    datapicker.register_callback(dataset_changed)
    return datapicker
