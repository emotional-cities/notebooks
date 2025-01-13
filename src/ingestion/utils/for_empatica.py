# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              IMPORT LIBRARIES                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import matplotlib.pyplot as plt
import numpy as np
from pluma.preprocessing.ecg import heartrate_from_ecg
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

def empatica_and_ecg_to_csv(datapicker, outdir):
    
    # Get LSL markers
    lsl_markers = datapicker.streams.EEG.server_lsl_marker[datapicker.streams.EEG.server_lsl_marker.MarkerIdx>35000]

    try:
        # Get heartrate from ECG (ERROR -> TypeError: 'DotMap' object is not callable)
        ecg_hr = heartrate_from_ecg(ecg = datapicker.streams.BioData.ECG,
                                    sample_rate = 50, bpmmax = 200)
    except:
        print('ERROR: heartrate_from_ecg')
    
    
    # Save to csv
    lsl_markers.to_csv(outdir+r'\lsl_markers.csv')
    try:
        ecg_hr.to_csv(outdir+r'\ecg_hr.csv')
        datapicker.streams.BioData.ECG.data.Value0.to_frame().to_csv(outdir+r'\ecg.csv')
    except:
        print('ERROR: error saving ecg_hr and ecg')
    datapicker.streams.Empatica.data.E4_Gsr.to_csv(outdir+r'\e4_gsr.csv')
    datapicker.streams.Empatica.data.E4_Temperature.to_csv(outdir+r'\e4_temp.csv')
    datapicker.streams.Empatica.data.E4_Ibi.to_csv(outdir+r'\e4_ibi.csv')
    datapicker.streams.Empatica.data.E4_Bvp.to_csv(outdir+r'\e4_bvp.csv')
    datapicker.streams.Empatica.data.E4_Acc.to_csv(outdir+r'\e4_acc.csv')
    datapicker.streams.Empatica.data.E4_Hr.to_csv(outdir+r'\e4_hr.csv')

    # output
    print('Data saved to:', outdir)
    print('Created files:')
    print('lsl_markers.csv')
    print('ecg_hr.csv')
    print('ecg.csv')
    print('e4_gsr.csv')
    print('e4_temp.csv')
    print('e4_ibi.csv')
    print('e4_bvp.csv')
    print('e4_acc.csv')
    print('e4_hr.csv')
    
    return

def export_resampled_empatica_data(input_dir, output_dir):
    """
    Processes EDA and related physiological data from CSV files,
    resamples to 1Hz, and saves the combined data to a CSV file.
    All generated figures are saved to the output directory.

    Parameters:
    - input_dir: Directory containing the input CSV files.
    - output_dir: Directory where the output CSV and figures will be saved.

    Output:
    - Saves 'data_all_1Hz.csv' to the output directory.
    - Saves all generated figures to the output directory.
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set default plotting parameters
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.rcParams.update({'font.size': 18})

    # Sampling frequencies
    fs_bvp = 64
    fs_eda = 4

    # Load data from CSV files (excluding ecg_hr.csv and ecg.csv)
    print("Loading data...")
    bvp_subj = pd.read_csv(os.path.join(input_dir, 'e4_bvp.csv'))
    ibi_subj = pd.read_csv(os.path.join(input_dir, 'e4_ibi.csv'))
    eda_subj = pd.read_csv(os.path.join(input_dir, 'e4_gsr.csv'))
    hr_subj = pd.read_csv(os.path.join(input_dir, 'e4_hr.csv'))
    acc_subj = pd.read_csv(os.path.join(input_dir, 'e4_acc.csv'))
    temp_subj = pd.read_csv(os.path.join(input_dir, 'e4_temp.csv'))

    # Process datetime columns
    print("Processing datetime columns...")
    eda_subj['DateTime'] = pd.to_datetime(eda_subj['E4_Seconds'].str[:-3], format="%Y-%m-%d %H:%M:%S.%f")
    bvp_subj['DateTime'] = pd.to_datetime(bvp_subj['E4_Seconds'].str[:-3], format="%Y-%m-%d %H:%M:%S.%f")
    ibi_subj['DateTime'] = pd.to_datetime(ibi_subj['E4_Seconds'].str[:-3], format="%Y-%m-%d %H:%M:%S.%f")
    hr_subj['DateTime'] = pd.to_datetime(hr_subj['E4_Seconds'].str[:-3], format="%Y-%m-%d %H:%M:%S.%f")
    acc_subj['DateTime'] = pd.to_datetime(acc_subj['E4_Seconds'].str[:-3], format="%Y-%m-%d %H:%M:%S.%f")
    temp_subj['DateTime'] = pd.to_datetime(temp_subj['E4_Seconds'].str[:-3], format="%Y-%m-%d %H:%M:%S.%f")

    # Rename columns for consistency
    eda_subj.rename(columns={'Value': 'Values'}, inplace=True)
    bvp_subj.rename(columns={'Value': 'Values'}, inplace=True)
    ibi_subj.rename(columns={'Value': 'Values'}, inplace=True)
    hr_subj.rename(columns={'Value': 'Values'}, inplace=True)
    acc_subj.rename(columns={'Value': 'Values'}, inplace=True)
    temp_subj.rename(columns={'Value': 'Values'}, inplace=True)

    # Compute accelerometer magnitude
    acc_subj['Magnitude'] = np.sqrt(acc_subj['AccX']**2 + acc_subj['AccY']**2 + acc_subj['AccZ']**2)

    # Filter EDA signal
    print("Filtering EDA signal...")
    eda_subj['Filtered'] = scipy.signal.savgol_filter(eda_subj['Values'], window_length=11, polyorder=5)
    b, a = scipy.signal.butter(5, 0.05, btype='highpass', fs=fs_eda)
    eda_subj['NEW_EDA'] = scipy.signal.filtfilt(b, a, eda_subj['Filtered'])

    # Plot EDA signals
    print("Plotting EDA signals...")
    plt.figure(figsize=(15, 7))
    plt.plot(eda_subj['DateTime'], eda_subj['Values'], label='Raw EDA')
    plt.plot(eda_subj['DateTime'], eda_subj['Filtered'], label='Filtered EDA')
    plt.plot(eda_subj['DateTime'], eda_subj['NEW_EDA'], label='High-pass Filtered EDA')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('EDA Value')
    plt.title('EDA Signal Processing')
    plt.savefig(os.path.join(output_dir, 'eda_signals.png'), dpi=300)
    plt.close()

    # Process IBI data
    print("Processing IBI data...")
    ibi_subj['IBI'] = ibi_subj['Values']
    ibi_subj['bpm'] = 60 / ibi_subj['IBI']

    # Plot HR signals (from IBI and E4 HR)
    print("Plotting HR signals...")
    plt.figure(figsize=(15, 7))
    plt.plot(hr_subj['DateTime'], hr_subj['Values'], label='E4 HR')
    plt.plot(ibi_subj['DateTime'], ibi_subj['bpm'], label='E4 IBI Heart Rate')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Heart Rate (bpm)')
    plt.title('Heart Rate Comparison (E4 HR and E4 IBI)')
    plt.savefig(os.path.join(output_dir, 'hr_signals.png'), dpi=300)
    plt.close()

    # Resample data to 1Hz
    print("Resampling data to 1Hz...")

    # Function to resample numeric columns only (change here the fs)
    def resample_numeric(df, datetime_col='DateTime', freq='1S'):
        df_numeric = df.select_dtypes(include=[np.number])
        df_numeric[datetime_col] = df[datetime_col]
        df_numeric = df_numeric.set_index(datetime_col).resample(freq).mean().reset_index()
        return df_numeric

    data_ibi = resample_numeric(ibi_subj)
    data_hr = resample_numeric(hr_subj)
    data_temp = resample_numeric(temp_subj)
    data_eda = resample_numeric(eda_subj)
    data_acc = resample_numeric(acc_subj)
    data_bvp = resample_numeric(bvp_subj)

    # Merge all data into a single DataFrame
    print("Merging data...")
    data_all = data_hr[['DateTime', 'Values']].rename(columns={'Values': 'E4_HR'})
    data_all = pd.merge(data_all, data_ibi[['DateTime', 'bpm']].rename(columns={'bpm': 'E4_HR_IBI'}), on='DateTime', how='outer')
    data_all = pd.merge(data_all, data_temp[['DateTime', 'Values']].rename(columns={'Values': 'TEMP'}), on='DateTime', how='outer')
    data_all = pd.merge(data_all, data_eda[['DateTime', 'Values', 'NEW_EDA']].rename(columns={'Values': 'EDA_RAW', 'NEW_EDA': 'EDA_PHASIC'}), on='DateTime', how='outer')
    data_all = pd.merge(data_all, data_acc[['DateTime', 'AccX', 'AccY', 'AccZ', 'Magnitude']], on='DateTime', how='outer')
    data_all = pd.merge(data_all, data_bvp[['DateTime', 'Values']].rename(columns={'Values': 'BVP_Values'}), on='DateTime', how='outer')

    # Plot merged data
    print("Plotting merged data...")
    plt.figure(figsize=(15, 7))
    plt.plot(data_all['DateTime'], data_all['E4_HR'], label='E4 HR')
    plt.plot(data_all['DateTime'], data_all['E4_HR_IBI'], label='E4 HR from IBI')
    plt.plot(data_all['DateTime'], data_all['EDA_RAW'], label='EDA Raw')
    plt.plot(data_all['DateTime'], data_all['TEMP'], label='Temperature')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Merged Physiological Data')
    plt.savefig(os.path.join(output_dir, 'merged_data.png'), dpi=300)
    plt.close()

    # Save combined data to CSV
    print("Saving combined data to CSV...")
    data_all.to_csv(os.path.join(output_dir, 'data_all_1Hz.csv'), index=False)

    print("Processing complete. All figures and CSV file saved to the output directory.")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                          PLOTTING FUNCTIONS                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
