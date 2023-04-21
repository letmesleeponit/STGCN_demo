import os, sys, pathlib
import cv2
import mediapipe as mp
from collections import deque
import numpy as np
import torch
sys.path.insert(1, "/home/jerry/st-gcn_ori") # Your STGCN dirpath
from processor.io import IO

### Transform the output format of mp to the one we are accustomed 
# results_hand : mediapipe hands solution result
# results_body : mediapipe body solution result
# dimension : output features dimension
def extract_pts(results_hand, results_body) -> dict:
    point_cor = {'pose_keypoints_2d':[], 'hand_left_keypoints_2d':[], 'hand_right_keypoints_2d':[]}
    hand_counter = 0 # count = 1 means left hand, 2 means right hand
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            hand_counter += 1 # use hand_counter to determine these points are represent the coordinate of which hand
            if hand_counter == 1: 
                hand = 'hand_left_keypoints_2d'
            elif hand_counter == 2:
                hand = 'hand_right_keypoints_2d'
            for hand_landmark in hand_landmarks.landmark:
                point_cor[hand].append(hand_landmark.x)
                point_cor[hand].append(hand_landmark.y)
    if results_body.pose_landmarks:
        for pose_landmark in results_body.pose_landmarks.landmark:
            point_cor['pose_keypoints_2d'].append(pose_landmark.x)
            point_cor['pose_keypoints_2d'].append(pose_landmark.y)
    return point_cor

### Transform the output format from what we are accustomed to a specific
# data : output from extract_pts function
def transformator(data:dict) -> dict:
    coordinates = []
    skeleton = {}

    for part in np.array(data.get("hand_right_keypoints_2d")).reshape(21, -1):
        x, y = part[0], part[1]
        coordinates += [x, y]
    for part in np.array(data.get("hand_left_keypoints_2d")).reshape(21, -1):
        x, y = part[0], part[1]
        coordinates += [x, y]
    arm = np.array(data.get("pose_keypoints_2d")).reshape(33, -1)
    for i in [0, 11, 12, 13, 14, 15, 16]:
        x, y = arm[i, 0], arm[i, 1]
        coordinates += [x, y]

    skeleton["pose"] = coordinates
    return skeleton

### Run the STGCN model
# cha_size : dimension
# skele_size : features number from skeleton
# P : STGCN class(from processor.IO)
# storae : keep the skeleton data within the window size
def st_gcn_hand(cha_size:int, skele_size:int, p:IO, storage:deque):
    p.model.eval()
    num_person_in = 1
    ### read the skeleton data
    data_numpy = np.zeros((cha_size, len(storage["data"]), skele_size, num_person_in))
    for frame_info in storage["data"]:
        frame_index = frame_info["frame_index"]
        for m, skeleton_info in enumerate(frame_info["skeleton"]):
            if m >= num_person_in:
                break
            pose = skeleton_info["pose"]
            for channel in range(cha_size):
                data_numpy[channel, frame_index, :, m] = pose[channel::cha_size]
    pose = data_numpy
    data = torch.from_numpy(pose)
    data = data.unsqueeze(0)
    data = data.float().to(p.dev).detach()

    output, feature = p.model.extract_feature(data) # Forward propagation
    output = output[0]
    feature = feature[0]
    intensity = (feature * feature).sum(dim=0) ** 0.5
    intensity = intensity.cpu().detach().numpy()

    label_name_path = os.path.join(pathlib.Path(__file__).parent, 'config/label.txt')
    with open(label_name_path) as f:
        label_name = f.readlines()
        label_name = [line.rstrip() for line in label_name]
    label_sequence = output.sum(dim=3).sum(dim=2).argmax(dim=0)
    label_name_sequence = [label_name[l] for l in label_sequence]
    label_recover = []
    for i in label_name_sequence:
        for _ in range(4):
            label_recover.append(i)
    label_predict = label_recover[: len(storage["data"])]
    return label_predict[-1]


### mediapipe parameter setting
vid_path = os.path.join(pathlib.Path(__file__).parent, 'vid.mp4')
cap = cv2.VideoCapture(vid_path)
mp_hands = mp.solutions.hands # mediapipe hands solution
mp_pose = mp.solutions.pose # mediapipe pose solution
point_coor = dict() # keep the skeleton coordinates in each frame

### STGCN parameter setting
storage = deque() # keep the skeleton data within the window size
max_frame = 40 # window size 
dimension, skele_size = 2, 49 # skeleton feature dimension, features number from skeleton
yaml_path = os.path.join(pathlib.Path(__file__).parent, 'model/demo.yaml')
argv = ['--config', yaml_path]
p = IO(argv) # build the STGCN object

### video parameter(write an video record the result)
inWidth = 1920
inHeight = 1080
vid_path = os.path.join(pathlib.Path(__file__).parent, 'output.mp4')
vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (inWidth,inHeight))
cv2.namedWindow('real_time', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('real_time', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

### Use mediapipe extract the skeleton feature
with mp_hands.Hands(
static_image_mode = False,
model_complexity = 0,
min_detection_confidence = 0.5,
min_tracking_confidence = 0.5) as hands, \
mp_pose.Pose(
min_detection_confidence=0.5,
min_tracking_confidence=0) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("The video does not exist.")
            break

        ## To improve performance, optionally mark the image as not writeable to
        ## pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hand = hands.process(image)
        results_body = pose.process(image)
        point_coor = extract_pts(results_hand, results_body)
        
        ### 
        data = point_coor
        storage_main = {'data':[]}
        # If the data has NaN value skip it directly
        if np.any([len(data[ind]) == 0 for ind in data.keys()]):
            continue
        storage.append(data)
        
        if len(storage) >= max_frame: 
            for frame in range(len(storage)):
                storage_main['data'].append({"frame_index":frame,"skeleton":[transformator(storage[frame])]})
            # inference STGCN model
            predict_label = st_gcn_hand(dimension, skele_size, p, storage_main)
            storage.popleft()
        
            ### 視覺化
            # Connection relationship between pose and hand
            POSE_PAIRS = [[11,12],[12,14],[14,16],[16,18],[16,20],[18,20],[16,22],[11,13],[13,15],[15,21],[15,17],[15,19],[17,19]]
            POSE_PAIRS_HAND = [[1,0],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[0,17],[13,17],[17,18],[18,19],[19,20]]
            
            string1 = predict_label
            data = storage[-1] # Skeleton information at last frame
            frame_draw = np.full((inHeight, inWidth, 3), 0)
            frame_draw = frame_draw.astype(np.uint8)
            pre_data = storage_main['data'][-1]['skeleton'][0]['pose']

            # Draw Skeleton(not prepocessing, so not contain 0~4)
            X = np.dot([data['pose_keypoints_2d'][i] for i in range(0,len(data['pose_keypoints_2d']),2)], inWidth)
            Y = np.dot([data['pose_keypoints_2d'][i] for i in range(1,len(data['pose_keypoints_2d']),2)], inHeight)

            # Draw Skeleton for left hand
            XL = np.dot([data['hand_left_keypoints_2d'][i] for i in range(0,len(data['hand_left_keypoints_2d']),2)], inWidth)
            YL = np.dot([data['hand_left_keypoints_2d'][i] for i in range(1,len(data['hand_left_keypoints_2d']),2)], inHeight)
            XR = np.dot([data['hand_right_keypoints_2d'][i] for i in range(0,len(data['hand_right_keypoints_2d']),2)], inWidth)
            YR = np.dot([data['hand_right_keypoints_2d'][i] for i in range(1,len(data['hand_right_keypoints_2d']),2)], inHeight)
            
            for i in POSE_PAIRS:
                point1 = i[0]
                point2 = i[1]
                cv2.line(frame_draw, (int(X[point1]),int(Y[point1])), (int(X[point2]),int(Y[point2])), (255, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame_draw, (int(X[point1]),int(Y[point1])), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame_draw, (int(X[point2]),int(Y[point2])), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        
            for i in POSE_PAIRS_HAND:
                point1 = i[0]
                point2 = i[1]
                # right hand
                cv2.line(frame_draw, (int(XR[point1]),int(YR[point1])), (int(XR[point2]),int(YR[point2])), (255, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame_draw, (int(XR[point1]),int(YR[point1])), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame_draw, (int(XR[point2]),int(YR[point2])), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                # left hand
                cv2.line(frame_draw, (int(XL[point1]),int(YL[point1])), (int(XL[point2]),int(YL[point2])), (255, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame_draw, (int(XL[point1]),int(YL[point1])), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame_draw, (int(XL[point2]),int(YL[point2])), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                
            cv2.putText(frame_draw,('Therblig:' + str(predict_label)),(40,40),cv2.FONT_HERSHEY_SIMPLEX,1.5, (30, 255, 255), 3)
            # paste all the therblig labels on the screen
            paste_therblig_labels = ['1.Reach', '2.Grasp', '3.Move', '4.Assembqle', '5.Use', '6.Position', '7.Release']
            paste_count = 0
            cv2.putText(frame_draw,('Therblig labels:'),(30, 300 + 40 * -1),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            for paste_therblig_label in paste_therblig_labels:
                cv2.putText(frame_draw,(str(paste_therblig_label)),(30, 300 + 40 * paste_count),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                paste_count += 1
            paste_count = 0
            
            vid_writer.write(frame_draw)
            cv2.imshow('real_time', frame_draw)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):        
                vid_writer.release()
                cv2.destroyAllWindows()
                exit() 
