# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              IMPORT LIBRARIES                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import biosppy
import os
import datetime

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                          PROCESSING FUNCTIONS                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import pandas as pd
import numpy as np

def do_analysis_design(results_dir, design, **kwargs):
    """
    Join 'alldata.csv' files from {results_dir}/sub-{subject}/ses-{session}/ 
    based on the 'design'.

    Args:
        results_dir (str): Directory where the 'fulldata' results are stored,
                           i.e. {results_dir}/sub-{participant}/ses-{session}/alldata.csv
        design (str): The analysis design:
            - 'within_subject': returns only the file from a single subject & session
            - 'between_subject': returns data from all subjects across all sessions
            - 'between_session': returns data from all sessions for one subject
            - 'within_path': returns data from all subjects for a single session (path)
            - 'between_path': for a list of session names (paths), compute nan-mean 
                              for each unique GPS coordinate across all subjects,
                              then return the concatenated data of those sessions.
        **kwargs: Additional arguments needed by each design:
            - subject (str): e.g. "OE001" or "OE022"
            - session (str or list): e.g. "Lapa" or ["Lapa","Graca"]
              (for 'between_path', we expect a list of sessions)
    Returns:
        pd.DataFrame: A DataFrame that meets the design specification 
                      (possibly empty if no data found).
    """

    # We'll store the final DataFrame
    combined_df = pd.DataFrame()

    def read_alldata_csv(subject, session):
        """
        Attempt to read the file {results_dir}/sub-{subject}/ses-{session}/alldata.csv
        Returns a DataFrame if found, else None.
        """
        fpath = os.path.join(results_dir, f"sub-{subject}", f"ses-{session}", "alldata.csv")
        if os.path.exists(fpath):
            try:
                return pd.read_csv(fpath)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                return None
        return None

    def list_all_sub_ses_pairs():
        """
        Explore {results_dir}/sub-???/ses-??? for existing alldata.csv.
        Returns a list of tuples (subject, session).
        """
        sub_ses_pairs = []
        for subdir in os.listdir(results_dir):
            if subdir.startswith("sub-"):
                subject_id = subdir[4:]  # e.g. "OE101"
                subdir_path = os.path.join(results_dir, subdir)
                if os.path.isdir(subdir_path):
                    for sesdir in os.listdir(subdir_path):
                        if sesdir.startswith("ses-"):
                            session_id = sesdir[4:]  # e.g. "Lapa"
                            # Check if alldata.csv exists
                            f = os.path.join(subdir_path, sesdir, "alldata.csv")
                            if os.path.isfile(f):
                                sub_ses_pairs.append((subject_id, session_id))
        return sub_ses_pairs

    if design == 'within_subject':
        subject = kwargs.get('subject', None)
        session = kwargs.get('session', None)
        if not subject or not session:
            print("Error: 'within_subject' design needs 'subject' and 'session'.")
            return pd.DataFrame()

        df = read_alldata_csv(subject, session)
        if df is not None:
            combined_df = df
        else:
            print(f"No data found for sub-{subject}, ses-{session}.")
        return combined_df

    elif design == 'between_subject':
        sub_ses_pairs = list_all_sub_ses_pairs()
        if not sub_ses_pairs:
            print("No sub-ses pairs found in results_dir.")
            return pd.DataFrame()

        dataframes = []
        for (subj, sess) in sub_ses_pairs:
            df = read_alldata_csv(subj, sess)
            if df is not None:
                dataframes.append(df)
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    elif design == 'between_session':
        subject = kwargs.get('subject', None)
        if not subject:
            print("Error: 'between_session' design needs 'subject'.")
            return pd.DataFrame()

        sub_ses_pairs = list_all_sub_ses_pairs()
        dataframes = []
        for (subj, sess) in sub_ses_pairs:
            if subj == subject:
                df = read_alldata_csv(subj, sess)
                if df is not None:
                    dataframes.append(df)

        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    elif design == 'within_path':
        session = kwargs.get('session', None)
        if not session:
            print("Error: 'within_path' design needs 'session'.")
            return pd.DataFrame()

        sub_ses_pairs = list_all_sub_ses_pairs()
        dataframes = []
        for (subj, sess) in sub_ses_pairs:
            if sess == session:
                df = read_alldata_csv(subj, sess)
                if df is not None:
                    dataframes.append(df)

        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    elif design == 'between_path':
        # We'll expect session to be a list of session names
        session_list = kwargs.get('session', None)
        if not session_list or not isinstance(session_list, (list, tuple)):
            print("Error: 'between_path' design requires 'session' to be a list of session names.")
            return pd.DataFrame()

        sub_ses_pairs = list_all_sub_ses_pairs()
        if not sub_ses_pairs:
            print("No sub-ses pairs found in results_dir.")
            return pd.DataFrame()

        # We'll store partial results in a list
        results_for_each_session = []

        for desired_session in session_list:
            # Gather all data from sub-ses pairs for that session
            dataframes = []
            for (subj, sess) in sub_ses_pairs:
                if sess == desired_session:
                    df = read_alldata_csv(subj, sess)
                    if df is not None and not df.empty:
                        dataframes.append(df)

            if not dataframes:
                print(f"No data found for session/path={desired_session}. Skipping.")
                continue

            # Concatenate all subject data for this path
            big_df = pd.concat(dataframes, ignore_index=True)

            # Group by (longitude, latitude) and compute the nan-mean of numeric columns
            if ('longitude' not in big_df.columns) or ('latitude' not in big_df.columns):
                print(f"No 'longitude' or 'latitude' in data for {desired_session}. Storing raw.")
                big_df['session_path'] = desired_session
                results_for_each_session.append(big_df)
                continue

            # Convert columns to numeric where possible
            numeric_cols = big_df.select_dtypes(include=[np.number]).columns.tolist()

            # Transform geometry to longitude and latitude columns
            if 'geometry' in big_df.columns:
                big_df['longitude'] = big_df['geometry'].apply(lambda x: x.x)
                big_df['latitude'] = big_df['geometry'].apply(lambda x: x.y)

            # Group by GPS coordinate
            grouped = (
                big_df.groupby(['longitude','latitude'], as_index=False)[numeric_cols]
                    .mean(numeric_only=True)
            )

            # Insert the session path label
            grouped['session_path'] = desired_session

            results_for_each_session.append(grouped)

        if results_for_each_session:
            combined_df = pd.concat(results_for_each_session, ignore_index=True)
        return combined_df

    else:
        print(f"Unknown design type: {design}")
        return pd.DataFrame()
    


### PLOT ###

def plot_mapped_data(shpdata, session):
    
    # Import necessary functions
    from utils import fetch_path_num
    
    # Get path information
    path_num = fetch_path_num(session)
    path_num = str(path_num).zfill(2) # make it a two-digit string

    # Get shapefile name
    if path_num == '01':
        shp_filename = "01_belem.shp"
    elif path_num == '02':
        shp_filename = "02_lapa.shp"
    elif path_num == '03':
        shp_filename = "03_gulbenkian.shp"
    elif path_num == '04':
        shp_filename = "04_Baixa.shp"
    elif path_num == '05':
        shp_filename = "05_Graca.shp"
    elif path_num == '06':
        shp_filename = "06_Pnacoes.shp"
    elif path_num == '07':
        shp_filename = "07_ANovas_Sa_Bandeira.shp"
    elif path_num == '08':
        shp_filename = "08_ANovas_CMoeda.shp"
    elif path_num == '09':
        shp_filename = "09_PFranca_Escolas.shp"
    elif path_num == '10':
        shp_filename = "10_PFranca_Morais_Soares.shp"
    elif path_num == '11':
        shp_filename = "11_Marvila_Beato.shp"
    elif path_num == '12':
        shp_filename = "12_PNacoes_Gare.shp"
    elif path_num == '13':
        shp_filename = "13_Madredeus.shp"
    elif path_num == '14':
        shp_filename = "14_Benfica_Pupilos.shp"
    elif path_num == '15':
        shp_filename = "15_Benfica_Moinhos.shp"
    elif path_num == '16':
        shp_filename = "16_Benfica_Grandella.shp"
    elif path_num == '17':
        shp_filename = "17_Restauradores.shp"
    elif path_num == '18':
        shp_filename = "18_Belem_Estadio.shp"
    elif path_num == '19':
        shp_filename = "19_Estrela_Jardim.shp"
    elif path_num == '20':
        shp_filename = "20_Estrela_Assembleia.shp"
    elif path_num == '21':
        shp_filename = "21_Estrela_Rato.shp"
    elif path_num == '22':
        shp_filename = "22_Estrela_Prazeres.shp"
    # Correct GPS data
    shp_file        = os.path.join(shpdata, shp_filename)

