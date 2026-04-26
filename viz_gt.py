import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R

filename = 'img000051.jpg'

# ---------- PATHS ----------
img_path = f"D:\\MLR\\speedplusv2\\speedplusv2\\synthetic\\images\\{filename}"
json_path = "D:\\MLR\\speedplusv2\\speedplusv2\\synthetic\\train.json"
mat_path = "D:\\MLR\\speedplusv2\\speedplusv2\\tangoPoints.mat"

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------- CAMERA ----------
K = np.array([
    [2988.5795163815555, 0, 960],
    [0, 2988.3401159176124, 600],
    [0, 0, 1]
])

# ---------- LOAD POINT CLOUD ----------
data = loadmat(mat_path)
points_3D = data['tango3Dpoints']  # (3, N)

# ---------- LOAD ANNOTATION ----------
with open(json_path) as f:
    annotations = json.load(f)

item = next(x for x in annotations if x["filename"] == filename)

q = item["q_vbs2tango_true"]
t = np.array(item["r_Vo2To_vbs_true"])

# ---------- ROTATION ----------
q_scipy = [q[1], q[2], q[3], q[0]]
R_mat = R.from_quat(q_scipy).as_matrix()

# =====================================================
# 3D BBOX
# =====================================================
xmin, ymin, zmin = points_3D.min(axis=1)
xmax, ymax, zmax = points_3D.max(axis=1)

bbox_3D = np.array([
    [xmin, ymin, zmin],
    [xmax, ymin, zmin],
    [xmax, ymax, zmin],
    [xmin, ymax, zmin],
    [xmin, ymin, zmax],
    [xmax, ymin, zmax],
    [xmax, ymax, zmax],
    [xmin, ymax, zmax],
])

# =====================================================
# TRANSFORM
# =====================================================
points_cam = (R_mat @ bbox_3D.T).T + t

# 🔥 FIX: don't drop points
points_cam[:, 2] = np.clip(points_cam[:, 2], 1e-6, None)

# =====================================================
# PROJECT
# =====================================================
points_2D = (K @ points_cam.T).T
points_2D = points_2D[:, :2] / points_2D[:, 2:]

# =====================================================
# PLOT
# =====================================================
fig, ax = plt.subplots(figsize=(8,6))
ax.imshow(img)

# edges
edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

color = 'lime'  # pick any: 'red', 'blue', 'white', '#00FF00'

# draw edges
for i, j in edges:
    x = [points_2D[i,0], points_2D[j,0]]
    y = [points_2D[i,1], points_2D[j,1]]
    ax.plot(x, y, color=color, linewidth=2)

# draw points
ax.scatter(points_2D[:,0], points_2D[:,1], color=color, s=30)

ax.set_title("3D Bounding Box")
ax.axis('off')

plt.show()