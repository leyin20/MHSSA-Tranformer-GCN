# ============================
# prepare_data.py
# ============================
"""数据准备与公共工具函数。"""
import os
import re
import cv2
import dlib
import numpy as np
import pandas as pd
from PIL import Image

# ----- 区域 & 常量 -----
REGION_LIST = ["left_eyebrow", "right_eyebrow", "left_eye", "right_eye", "nose", "outer_lip"]
REGION_KEYPOINTS = {
    "left_eyebrow":  [4, 3, 2, 1, 0],
    "right_eyebrow": [5, 6, 7, 8, 9],
    "left_eye":      [10, 11, 12, 15, 14, 13],
    "right_eye":     [19, 18, 17, 21, 20, 16],
    "nose":          [22, 23, 24, 25, 28, 29, 27, 30, 26],
    "outer_lip":     [37, 31, 38, 36, 42, 32, 39, 35, 41, 33, 40, 34],
}
REGION_PATCH_COUNTS = {"left_eyebrow":5,"right_eyebrow":5,"left_eye":6,"right_eye":6,"nose":9,"outer_lip":12}
REGION_ID_DICT = {r:i for i,r in enumerate(REGION_LIST)}
REGION_AU_DICT = {
    "left_eyebrow":[1,2,4],"right_eyebrow":[1,2,4],
    "left_eye":[5,6,7],"right_eye":[5,6,7],
    "nose":[9],"outer_lip":[10,25]
}

# ----- AU 邻接矩阵 -----
def evaluate_adj(csv_path:str)->np.ndarray:
    df = pd.read_csv(csv_path)
    num_r=len(REGION_LIST)
    count=np.zeros((num_r,num_r)); reg=np.zeros(num_r)
    for units in df["Action Units"].astype(str):
        au=list(map(int,re.findall(r"\d+",units)))
        active=np.zeros(num_r)
        for region,aus in REGION_AU_DICT.items():
            if set(au)&set(aus): active[REGION_ID_DICT[region]]=1
        for i in range(num_r):
            if active[i]:
                reg[i]+=1
                for j in range(i+1,num_r):
                    if active[j]: count[i,j]+=1;count[j,i]+=1
    reg[reg==0]=1
    return count/reg.reshape(-1,1)

# ----- dlib -----
CNN_MODEL="mmod_human_face_detector.dat";PRED_MODEL="shape_predictor_68_face_landmarks.dat"
cnn_det=dlib.cnn_face_detection_model_v1(CNN_MODEL)
shape_pred=dlib.shape_predictor(PRED_MODEL)

# ----- 全局关键点 -----
def whole_face_block_coordinates(meta_csv="combined_3_class2_for_optical_flow.csv",base_dir="./datasets/combined_datasets_whole"):
    coords={}
    for _,row in pd.read_csv(meta_csv).iterrows():
        img_p=os.path.join(base_dir,row["imagename"])
        img=cv2.imread(img_p) or cv2.cvtColor(np.array(Image.open(img_p)),cv2.COLOR_RGB2BGR)
        if img is None: continue
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=cnn_det(gray,1)
        if not faces: continue
        face=faces[0].rect if hasattr(faces[0],'rect') else faces[0]
        sp=shape_pred(gray,face)
        kp=[(sp.part(i).x//8,sp.part(i).y//8) for i in range(17,68)]
        coords[row["imagename"]]=kp
    return coords

# ----- 裁剪光流块 -----
def crop_optical_flow_block(flow_dir="./datasets/STSNet_whole_norm_u_v_os"):
    coords=whole_face_block_coordinates()
    patches_dict={};kp_dict={}
    for f in os.listdir(flow_dir):
        img=cv2.imread(os.path.join(flow_dir,f));
        if img is None or f not in coords: continue
        patches_dict[f];kp_dict[f]={}
        kps=coords[f]
        patches_dict[f]={};kp_dict[f]={}
        for r in REGION_LIST:
            ps=[];ks=[]
            for idx in REGION_KEYPOINTS[r]:
                x,y=kps[idx];x,y=np.clip(x,3,24),np.clip(y,3,24)
                patch=img[y-3:y+4,x-3:x+4]
                if patch.shape[:2]!=(7,7):patch=cv2.resize(patch,(7,7))
                ps.append(patch);ks.append((x,y))
            patches_dict[f][r]=ps;kp_dict[f][r]=ks
    return patches_dict,kp_dict

__all__=["REGION_LIST","REGION_KEYPOINTS","REGION_PATCH_COUNTS","REGION_ID_DICT","evaluate_adj","crop_optical_flow_block"]