import cv2
import pandas as pd


def create_gaze_tracking(pupil_labs_stream=None):
    frames = pupil_labs_stream.RawFrames.data.copy()
    frames['Valid'] = pupil_labs_stream.DecodedFrames.data.Value.tolist()

    frames.index.names = ['HarpTimestamp']
    frames.reset_index(inplace = True)
    frames.set_index('Timestamp',inplace=True)
    frames = frames[frames.Valid !=0]
    gaze_data = pupil_labs_stream.PupilGaze.data.copy()
    gaze_data.index.names = ['HarpTimestamp']
    gaze_data.reset_index(inplace = True)
    gaze_data.set_index('Timestamp',inplace=True)
    gaze_coorelation = pd.merge_asof(frames, gaze_data[['GazeX','GazeY']], left_index=True, right_index=True)
    gaze_coorelation.index.names = ['PupilTimestamp']
    gaze_coorelation.reset_index(inplace = True)
    gaze_coorelation.set_index('HarpTimestamp',inplace=True)
    gaze_coorelation

    return gaze_coorelation
    # cap = cv2.VideoCapture(path+'\\'+source_video)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    # size = (width, height)
    # if fourcc is None:
    #     fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # type: ignore
    # out = cv2.VideoWriter(path+'\\'+destination_video,fourcc, 20.0, size)
    # count =0
     
    # while (cap.isOpened()):
    #     ret, frame = cap.read()
    #     if(ret == True ):
    #         gaze = gaze_coorelation.iloc[[count]]
    #         center_coordinates = (int(gaze.GazeX.item()),int(gaze.GazeY.item()))
    #         color = (0, 0, 255)
    #         cv2.circle(frame, center_coordinates, 30, color, 2)
    #         out.write(frame)
    #         count = count+1
    #     else:
    #         break
    # cap.release()
    # out.release()

def create_pupil_video_slice(path=None, source_video=None, destination_video=None, start_time=None, end_time=None, pupil_labs_stream=None, fourcc=None, gazeTrackingStream=False):
    
    frames = pupil_labs_stream.RawFrames.data.copy()
    frames['Valid'] = pupil_labs_stream.DecodedFrames.data.Value.tolist()
    frames = frames[frames.Valid !=0]
    if (start_time == None):
        start_time = frames.iloc[0].name
    if(end_time == None):
        end_index = frames.iloc[-1].name
    if(gazeTrackingStream != False):
        gaze_coorelation = create_gaze_tracking(pupil_labs_stream=pupil_labs_stream)
        #         gaze = gaze_coorelation.iloc[[count]]
        #         center_coordinates = (int(gaze.GazeX.item()),int(gaze.GazeY.item()))
        #         color = (0, 0, 255)
        #         cv2.circle(frame, center_coordinates, 30, color, 2)
        #         out.write(frame)
        #         count = count+1
        #     else:
        #         break
        # cap.release()
        # out.release()
    
    timeslice = frames[(frames.index >= start_time) & (frames.index <=end_time)]
    total_frames = timeslice.shape[0]
    if (total_frames <=0):
        raise ValueError("Invalid Time interval")
    startTime = timeslice.iloc[0].name
    test = frames.index.get_loc(startTime)
    if isinstance(test,slice) :
        start_index = test.start
    else :
        start_index= test 
    
    print(start_index)
    print(total_frames)

    cap = cv2.VideoCapture(path+'\\'+source_video)
    if(cap.isOpened()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        frame_count =0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        size = (width, height)
        if fourcc is None:
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # type: ignore
        out = cv2.VideoWriter(path+'\\'+destination_video,fourcc, 20.0, size)
        while (cap.isOpened()): 
            ret, frame = cap.read()
            if(ret == True) and (frame_count < total_frames):
                if(gazeTrackingStream):
                    gaze = gaze_coorelation.iloc[[start_index+frame_count]]
                    center_coordinates = (int(gaze.GazeX.item()),int(gaze.GazeY.item()))
                    color = (0, 0, 255)
                    cv2.circle(frame, center_coordinates, 30, color, 2)
                out.write(frame)
                frame_count = frame_count+1 
            else:
                break
        out.release()
    cap.release()