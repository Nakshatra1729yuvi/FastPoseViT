import torch
import torch.nn as nn
from transformers import ViTModel

class FPV(nn.Module):
  def __init__(self):
    super().__init__()
    self.vit=ViTModel.from_pretrained('google/vit-base-patch16-224')
    embed_dim=self.vit.config.hidden_size
    self.fc=nn.Linear(embed_dim,9)

  def forward(self,x):
    x=self.vit(x)
    cls_token=x.last_hidden_state[:,0]
    return self.fc(cls_token)


model=FPV()

x=torch.randn([5,3,224,224])
print(model(x).shape)