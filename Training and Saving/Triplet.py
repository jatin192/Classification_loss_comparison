# File uses the preprocessing done by the ipynb file. Please run that first 
# to create a pkl file. The pkl file will be directly loaded here.

import os
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pickle
import torchvision
from tqdm.notebook import tqdm



# In[3]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[4]:


config = dict(
    saved_path="saved_models/densenet121_triplet.pt",
    lr=0.001, 
    EPOCHS = 70,
    BATCH_SIZE = 16,
    IMAGE_SIZE = 128,
    TRAIN_VALID_SPLIT = 0.2,
    device=device,
    SEED = 42,
    pin_memory=True,
    num_workers=3,
    USE_AMP = True,
    channels_last=False)

random.seed(config['SEED'])
np.random.seed(config['SEED'])
torch.manual_seed(config['SEED'])
torch.cuda.manual_seed(config['SEED'])


# In[5]:


torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


# In[6]:


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAutocontrast(0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# In[7]:


a_file = open("dataset.pkl", "rb")
dataset = pickle.load(a_file)


# In[8]:


print(len(set(dataset['labels'][:500])))


# In[9]:


class Custom_data(Dataset):
    def __init__(self, dataset, transform = data_transforms, train=True):
        super(Custom_data,self).__init__()
        self.train_transforms = transform['train']
        self.test_transforms = transform['test']
        self.is_train = train
        self.to_pil = transforms.ToPILImage()
        
        if self.is_train:
            self.images = dataset['images']
            self.labels = np.array(dataset['labels'])
            self.index = np.array(list(range(len(self.labels))))
        
        else:
            self.images = dataset['images']
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_img = self.images[item]
        
        if self.is_train:
            anchor_label = self.labels[item]
            positive_list = self.index[self.index!=item][self.labels[self.index!=item]==anchor_label]

            positive_item = random.choice(positive_list)
            positive_img = self.images[positive_item]

            negative_list = self.index[self.index!=item][self.labels[self.index!=item]!=anchor_label]
            negative_item = random.choice(negative_list)
            negative_img = self.images[negative_item]

            if self.train_transforms:
                anchor_img = self.train_transforms(self.to_pil(anchor_img))
                positive_img = self.train_transforms(self.to_pil(positive_img))
                negative_img = self.train_transforms(self.to_pil(negative_img))

                return anchor_img, positive_img, negative_img, anchor_label
        
        else:
            if self.transform:
                anchor_img = self.test_transforms(self.to_pil(anchor_img))
            return anchor_img


# In[10]:


train_ds = Custom_data(dataset, train=True)
train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['num_workers'])


# In[11]:


a = iter(train_loader)
b = next(a)
print(b[0].shape, b[1].shape, b[2].shape, b[3].shape)


def train(model,train_loader,criterion):
    model.train()
    for epoch in range(config['EPOCHS']):
        running_loss = []
        print('Start of Epoch',epoch+1)
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_loader):
            anchor_img = anchor_img.to(config['device'])
            positive_img = positive_img.to(config['device'])
            negative_img = negative_img.to(config['device'])

            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} -Triplet Loss: {:.3f}".format(epoch+1, config['EPOCHS'], np.mean(running_loss)))
    torch.save(model.state_dict(), config['saved_path'])

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
    
densenet = models.densenet121(pretrained = True)
for name,parameter in densenet.named_parameters():
    if name == 'features.denseblock2.denselayer5.conv1.weight':
        break
    parameter.requires_grad = False
    
densenet.classifier = Identity()
model = densenet
#print(model)
model = model.to(config['device'])

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.TripletMarginLoss(margin=25.0)


train(model,train_loader,criterion)


    

