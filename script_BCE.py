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
import matplotlib.gridspec as gridspec

import random
import warnings
warnings.filterwarnings("ignore")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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
			label = torch.tensor([0]).float()
		else:
			img1 = Image.open(self.img_paths[idx][0])
			img2 = Image.open(self.img_paths[idx][0])
			label = torch.tensor([1]).float()

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

train_dataset = PCBDataset(img_paths[:1000], transform=data_transform)
test_dataset = PCBDataset(img_paths[1000:1250], transform=data_transform)
val_dataset = PCBDataset(img_paths[1250:], transform=data_transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=32)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=32)

data_iter = iter(train_loader)
images, label = data_iter.next()
# plt.subplot(1, 2, 1)
# plt.imshow(images.numpy()[10, 0, :, :], cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(images.numpy()[10, 1, :, :], cmap='gray')
# plt.show()

# class ContrastiveLoss(torch.nn.Module):

#       def __init__(self, margin=2):
#             super(ContrastiveLoss, self).__init__()
#             self.margin = margin

#       def forward(self, output1, output2, label):
#             # Find the pairwise distance or eucledian distance of two output feature vectors
#             euclidean_distance = F.pairwise_distance(output1, output2)
#             # perform contrastive loss calculation with the distance
#             loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#             (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

#             return loss_contrastive

# def test(model,img1,img2):
#        # Gives you the feature vector of both inputs
#        output1,output2 = model(img1.cpu(),img2.cpu())
#        # Compute the distance 
#        euclidean_distance = F.pairwise_distance(output1, output2)
#        #with certain threshold of distance say its similar or not
#        if eucledian_distance > 0.5:
#                print("Orginal PCB")
#        else:
#                print("Defected PCB")


class SiameseNet(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(SiameseNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=6, padding=2, stride=2)
		self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
		self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)


		self.fc1 = nn.Linear(8*8*32, 1024)
		self.fc2 = nn.Linear(1024, out_channels)

	def forward_once(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x), inplace=True), 2)
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = F.max_pool2d(F.relu(self.conv3(x)), 2)
		x = x.reshape(-1, 8*8*32)
		x = F.sigmoid(self.fc1(x))
		return x

	def forward(self, x):
		x1 = x[:, 0, :, :].reshape(-1, 1, 128, 128)
		x2 = x[:, 1, :, :].reshape(-1, 1, 128, 128)
		x1 = self.forward_once(x1)
		x2 = self.forward_once(x2)
		dist = torch.abs(x1 - x2)
		out = self.fc2(dist)
		return out


def compute_accuracy(y_pred, y_target):
	y_target = y_target.cpu()
	y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu()
	n_correct = torch.eq(y_pred_indices, y_target).sum().item()
	return n_correct / len(y_pred_indices) * 100

net = SiameseNet(1, 1)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

n_epochs = 1
BATCH_SIZE = 32

for epoch in range(n_epochs):
	train_loss = 0
	train_acc = 0
	train_counter = 0
	net.train()
	for images, label in train_loader:
		optimizer.zero_grad()
		output = net(images)
		loss = criterion(output, label)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		acc = compute_accuracy(output, label)
		train_acc += acc
		train_counter += 1
	train_loss /= train_counter 
	train_acc /= train_counter

	val_loss = 0
	val_acc = 0
	val_counter = 0
	net.eval()
	for images, label in val_loader:
		output = net(images)
		loss = criterion(output, label)
		val_loss += loss.item()
		acc = compute_accuracy(output, label)
		val_acc += acc
		val_counter += 1
	val_loss /= val_counter 
	val_acc /= val_counter
	
	print('[Epoch {}/{}] -> Train Loss: {:.4f}, Train acc: {:.3f} Val Loss: {:.4f}, Val acc: {:.3f}'\
          .format(epoch+1, n_epochs, train_loss, train_acc, val_loss, val_acc))
	
	if (val_acc >= 99.0):
		break

torch.save(net.state_dict(), 'model.pth')

net = SiameseNet(1, 1)
net.load_state_dict(torch.load('model.pth'))

images, label = iter(test_loader).next()
output = net(images)
output = torch.sigmoid(output)>0.5


label_dict = {1: "normal", 0: "defected", True: "normal", False: "defected"}
# fig = plt.figure(figsize=(10, 8))
# outer = gridspec.GridSpec(4, 4)

# for i in range(16):
# 	inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i])
# 	for j in range(2):
# 		ax = plt.Subplot(fig, inner[j])

# 		ax.imshow(images.numpy()[i, j, :, :], cmap='gray')
# 		if (j==0):
# 			t = ax.text(1.0,0.5, str(round(output[i].item(), 2)))
# 			t.set_ha('center')
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		fig.add_subplot(ax)
# fig.show()
# plt.show()

for i in range(32):
	fig = plt.figure()
	ax=fig.add_subplot(2,1,1)        
	plt.imshow(images.numpy()[i, 0, :, :], cmap='gray')
	ax=fig.add_subplot(2,1,2)        
	plt.imshow(images.numpy()[i, 1, :, :], cmap='gray')
	fig.suptitle("predicted : "+str(label_dict[output[i].item()])+" actual : "+str(label_dict[label[i].item()])) # or plt.suptitle('Main title')
	plt.show()

