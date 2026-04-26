import json
import cv2
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
import csv
import os
from tqdm import tqdm

fieldnames=['img_name','x_min','x_max','y_min','y_max']
train_json="D:\\MLR\\speedplusv2\\speedplusv2\\synthetic\\validation.json"
img_folder="D:\\MLR\\speedplusv2\\speedplusv2\\synthetic\\images"
mat_path = "D:\\MLR\\speedplusv2\\speedplusv2\\tangoPoints.mat"


K=np.array([
    [2988.5795163815555, 0, 960],
    [0, 2988.3401159176124, 600],
    [0, 0, 1]
])

data = loadmat(mat_path)
points_3D = data['tango3Dpoints'].T

with open(train_json) as f:
    annotations = json.load(f)

ann_dict = {item["filename"]: item for item in annotations}


with open('D:\\MLR\\FastPoseVit\\Scripts\\bbox_val.csv','w') as f:
    writer=csv.writer(f)
    writer.writerow(fieldnames)
    for filename in tqdm(os.listdir(img_folder)):
        file_path=img_folder+f"\\{filename}"
        img=cv2.imread(file_path)
        item = ann_dict.get(filename, None)
        if item is None:
            continue
        q=item["q_vbs2tango_true"] 
        t=np.array(item["r_Vo2To_vbs_true"])
        q_scipy=[q[1],q[2],q[3],q[0]]
        R_mat=R.from_quat(q_scipy).as_matrix()
        
        points_cam=(R_mat@points_3D.T).T+t

        valid=points_cam[:,2]>0
        points_cam=points_cam[valid]

        points_2D=(K@points_cam.T).T
        points_2D=points_2D[:,:2]/points_2D[:,2:]

        x_min,y_min=points_2D.min(axis=0)
        x_max,y_max=points_2D.max(axis=0)

        x_min,y_min=int(max(0,x_min)),int(max(0,y_min))
        x_max,y_max=int(min(img.shape[1],x_max)),int(min(img.shape[0],y_max))
        row=[filename,x_min,x_max,y_min,y_max]
        writer.writerow(row)
