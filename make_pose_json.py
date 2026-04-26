import csv
import json
import os


csv_path = "D:\\MLR\\FastPoseVit\\Scripts\\bbox_train.csv"
speed_json_path = "D:\\MLR\\speedplusv2\\speedplusv2\\synthetic\\train.json"
output_json = "D:\\MLR\\FastPoseVit\\Scripts\\pose_train.json"

# load SPEED annotations (has q, T)
with open(speed_json_path) as f:
    speed_data = json.load(f)

# make fast lookup
speed_dict = {item["filename"]: item for item in speed_data}

pose_data = []

with open(csv_path) as f:
    reader = csv.DictReader(f)

    for row in reader:
        filename = row["img_name"]

        if filename not in speed_dict:
            continue

        item = speed_dict[filename]

        x_min = int(row["x_min"])
        x_max = int(row["x_max"])
        y_min = int(row["y_min"])
        y_max = int(row["y_max"])

        w = x_max - x_min
        h = y_max - y_min

        # 🔥 add margin
        margin = 0.1
        x_min = max(0, x_min - margin*w)
        y_min = max(0, y_min - margin*h)
        w = w * (1 + 2*margin)
        h = h * (1 + 2*margin)

        pose_data.append({
            "filename": filename,
            "x_min": x_min,
            "y_min": y_min,
            "w": w,
            "h": h,
            "q": item["q_vbs2tango_true"],
            "T": item["r_Vo2To_vbs_true"]
        })

with open(output_json, "w") as f:
    json.dump(pose_data, f, indent=2)


