def empatica_and_ecg_to_csv(datapicker, outdir):
    
    # Process datapicker
    lsl_markers = datapicker.dataset.streams.EEG.server_lsl_marker[datapicker.dataset.streams.EEG.server_lsl_marker.MarkerIdx>35000]
    ecg_hr,_ = heartrate_from_ecg(datapicker.dataset.streams.BioData.ECG,
                                         fs = 250, max_heartrate_bpm = 250.0,
                                        peak_height = 800, smooth_win = 10)
    
    # Save to csv
    lsl_markers.to_csv(outdir+r'\lsl_markers.csv')
    ecg_hr.to_csv(outdir+r'\ecg_hr.csv')
    datapicker.dataset.streams.Empatica.data.E4_Gsr.to_csv(outdir+r'\e4_gsr.csv')
    datapicker.dataset.streams.Empatica.data.E4_Temperature.to_csv(outdir+r'\e4_temp.csv')
    datapicker.dataset.streams.Empatica.data.E4_Ibi.to_csv(outdir+r'\e4_ibi.csv')
    datapicker.dataset.streams.Empatica.data.E4_Bvp.to_csv(outdir+r'\e4_bvp.csv')
    datapicker.dataset.streams.Empatica.data.E4_Acc.to_csv(outdir+r'\e4_acc.csv')
    datapicker.dataset.streams.Empatica.data.E4_Hr.to_csv(outdir+r'\e4_hr.csv')
    datapicker.dataset.streams.BioData.ECG.data.Value0.to_frame().to_csv(outdir+r'\ecg.csv')

