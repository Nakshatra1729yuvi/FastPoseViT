import torch
from torch.utils.data import Dataset,DataLoader
import json
import cv2
import numpy as np
from utils import get_target
from tqdm import tqdm



K=np.array([
    [2988.5795163815555, 0, 960],
    [0, 2988.3401159176124, 600],
    [0, 0, 1]
])



class PoseDatatset(Dataset):
  def __init__(self,json_path,img_dir):
    with open(json_path,'r') as f:
      self.data=json.load(f)
    self.img_dir=img_dir
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self,idx):
    item=self.data[idx]
    img_path=self.img_dir+"\\"+item['filename']
    img=cv2.imread(img_path)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    x_min=int(item['x_min'])
    y_min=int(item['y_min'])
    w=int(item['w'])
    h=int(item['h'])

    crop_img=img[y_min:y_min+h,x_min:x_min+w]

    crop_img=cv2.resize(crop_img,(224,224))
    crop_img=crop_img.astype(np.float32)
    crop_img=crop_img/255.0 
    crop_img=torch.from_numpy(crop_img)
    crop_img=crop_img.permute(2,0,1)

    

    target=get_target(item['q'],item['T'],x_min,y_min,w,h,K)
    crop_img=crop_img.to(torch.float16)
    target=torch.tensor(target,dtype=torch.float16)

    return crop_img,target

print("started")


ds=PoseDatatset(json_path="D:\\MLR\\FastPoseVit\\Scripts\\pose_val.json",img_dir="D:\\MLR\\speedplusv2\\speedplusv2\\synthetic\\images")

loader=DataLoader(dataset=ds,batch_size=512,shuffle=True)

imgs=[]
tgts=[]


for x,y in tqdm(loader):
    imgs.append(x)
    tgts.append(y)

images=torch.cat(imgs,dim=0)
targets=torch.cat(tgts,dim=0)

torch.save(images,'D:\\MLR\\FastPoseVit\\Scripts\\img_val.pth')
torch.save(targets,'D:\\MLR\\FastPoseVit\\Scripts\\tgt_val.pth')