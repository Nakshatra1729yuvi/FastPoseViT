import csv
import json
import cv2
import os
from tqdm import tqdm

csv_path="D:\\MLR\\FastPoseVit\\Scripts\\bbox_val.csv"
output_json="D:\\MLR\\FastPoseVit\\Scripts\\bbox_val.json"
img_folder="D:\\MLR\\speedplusv2\\speedplusv2\\synthetic\\images"

images=[]
annotations=[]
img_id=0
ann_id=0

with open(csv_path) as f:
    reader=csv.DictReader(f)

    for row in tqdm(list(reader)):
        filename=row["img_name"]
        img_path=img_folder+f"\\{filename}"
        img=cv2.imread(img_path)
        if img is None:
            continue
        
        h,w=img.shape[:2]

        x_min=int(row["x_min"])
        x_max=int(row["x_max"])
        y_min=int(row["y_min"])
        y_max=int(row["y_max"])

        og_w=x_max-x_min
        og_h=y_max-y_min

        d_w=0.1*og_w
        d_h=0.1*og_h

        x_min_og=max(0,x_min-d_w)
        y_min_og=max(0,y_min-d_h)
        x_max_og=min(w,x_max+d_w)
        y_max_og=min(h,y_max+d_h)

        bbox_w=x_max_og-x_min_og
        bbox_h=y_max_og-y_min_og

        images.append({
            "id": img_id,
            "file_name": filename,
            "width": w,
            "height": h
        })

        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "bbox": [x_min_og, y_min_og, bbox_w, bbox_h],
            "category_id": 1,
            "area": bbox_w * bbox_h,
            "iscrowd": 0
        })

        img_id+=1
        ann_id+=1

data={
    "images": images,
    "annotations": annotations,
    "categories": [
        {"id": 1, "name": "spacecraft"}
    ]
}


with open(output_json, "w") as f:
    json.dump(data, f)