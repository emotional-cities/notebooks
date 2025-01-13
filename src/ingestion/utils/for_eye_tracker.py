# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              IMPORT LIBRARIES                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
import os

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                          PROCESSING FUNCTIONS                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def export_gaze_to_csv(dataset, outdir):

    gaze_timestamps = dataset.streams.PupilLabs.Counter.Gaze.data
    #gaze_timestamps.reset_index(inplace = T1rue)
    gaze_data = dataset.streams.PupilLabs.PupilGaze.data
    gaze = gaze_timestamps.join(gaze_data, on='Value') 
    gaze = gaze.drop('Value', axis=1)
    video_frames = dataset.streams.PupilLabs.Counter.DecodedFrames.data
    video_frames = video_frames[video_frames.Value !=0]
    #eyetracking_correlation = pd.merge_asof(data, space_time, left_index=True, right_index=True)
    gaze_coorelation = pd.merge_asof(video_frames, gaze, left_index=True, right_index=True)
    gaze_coorelation.to_csv(os.path.join(outdir,'gaze.csv'))
    