import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import matplotlib.pyplot as plt
import random

def get_file_paths():
	# collect file paths
	img_paths = []

	for file1 in glob.glob('PCBData/*'):
		for file2 in glob.glob(file1+'/*'):
			if '_not' not in file2:
				files_list = sorted(glob.glob(file2+'/*'))
				for i in range(0, len(files_list), 2):
					file_path = (files_list[i].split("_"))[0]
					img_paths.append([file_path+"_temp.jpg", file_path+"_test.jpg"])
				
	return img_paths

class PCBDataset(Dataset):
	def __init__(self, img_paths, transform=None):
		self.img_paths = img_paths
		self.transform = transform
	
	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, idx):
		if random.random() > 0.5:
			# different
			img1 = Image.open(self.img_paths[idx][0])
			img2 = Image.open(self.img_paths[idx][1])
			label = 0
		else:
			img1 = Image.open(self.img_paths[idx][0])
			img2 = Image.open(self.img_paths[idx][0])
			label = 1

		if self.transform:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
		final = torch.cat((img1, img2), axis=0)
		
		return final, label

img_paths = get_file_paths()

data_transform = transforms.Compose([
	transforms.Resize(128),
	transforms.Grayscale(num_output_channels=1),
	transforms.ToTensor()])

dataset = PCBDataset(img_paths, transform=data_transform)

data_loader = DataLoader(dataset, shuffle=True, batch_size=32)

data_iter = iter(data_loader)
images, label = data_iter.next()
# plt.subplot(1, 2, 1)
# plt.imshow(images.numpy()[10, 0, :, :], cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(images.numpy()[10, 1, :, :], cmap='gray')
# plt.show()

class ContrastiveLoss(torch.nn.Module):

      def __init__(self, margin=2):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

      def forward(self, output1, output2, label):
            # Find the pairwise distance or eucledian distance of two output feature vectors
            euclidean_distance = F.pairwise_distance(output1, output2)
            # perform contrastive loss calculation with the distance
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            return loss_contrastive

def test(model,img1,img2):
       # Gives you the feature vector of both inputs
       output1,output2 = model(img1.cpu(),img2.cpu())
       # Compute the distance 
       euclidean_distance = F.pairwise_distance(output1, output2)
       #with certain threshold of distance say its similar or not
       if eucledian_distance > 0.5:
               print("Orginal PCB")
       else:
               print("Defected PCB")


class SiameseNet(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(SiameseNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=6, padding=2, stride=2)
		self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
		self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)


		self.fc1 = nn.Linear(8*8*32, 1024)
		self.fc2 = nn.Linear(1024, out_channels)

	def forward_once(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = F.max_pool2d(F.relu(self.conv3(x)), 2)
		x = x.view(-1, 8*8*32)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

	def forward(self, x):
		x1 = x[:, 0, :, :].reshape(-1, 1, 128, 128)
		x2 = x[:, 1, :, :].reshape(-1, 1, 128, 128)
		x1 = self.forward_once(x1)
		x2 = self.forward_once(x2)
		return x1, x2
net = SiameseNet(1, 1)

criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-2)

n_epochs = 2

for _ in range(n_epochs):
	loss_val = 0
	count = 0
	for images, label in data_loader:
		optimizer.zero_grad()
		output1, output2 = net(images)
		loss_contrastive = criterion(output1,output2,label)
		loss_contrastive.backward()
		optimizer.step()
		loss_val += loss.item()
		count += 1
	print (loss_val/(count))