'''

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "model_name.onnx",   		 # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})

# Note : Critic is not required to be exported. 
         Hence it is not exported.

'''

import onnx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from numpy.random import *
import sys
import time
import torch.onnx 
import torch.nn.functional as F

from onnx_tf.backend import prepare
import tensorflow as tf
from onnx2keras import onnx_to_keras
import glob
import random

from pathlib import Path

import warnings

warnings.filterwarnings("ignore")

def export2onnx(images, label, model):
    export_path = "model.onnx"
    pred = model(images)
    # torch.onnx.export(model, images, export_path, export_params=True, input_names=['main_input'], output_names=['main_output'])
    torch.onnx.export(model, images, export_path, export_params=True, input_names=['input'], output_names=['output'])

    size = Path(export_path).stat().st_size
    print ("PyTorch model exported to {} size - {} bytes".format(export_path, size))
    return export_path

def onnx2tf(onnx_path):
    export_path = "model.pb"
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model) 
    tf_rep.export_graph(export_path)
    size = Path(export_path).stat().st_size
    print ("ONNX model exported to {} size - {} bytes".format(export_path, size))
    return export_path

def tf2tflite(tf_path):
  export_path = "model.tflite"


  converter = tf.lite.TFLiteConverter.from_frozen_graph(tf_path,
                                                  input_arrays=['input'], # input arrays 
                                                  output_arrays=['add_12'],  # output arrays as told in upper in my model case it si add_10
                                                  input_shapes=None
                                                  )

  # tell converter which type of optimization techniques to use
  # converter.optimizations = [tf.lite.Optimize.DEFAULT]
  # to view the best option for optimization read documentation of tflite about optimization go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional
  converter.experimental_new_converter = True
  converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
  # convert the model 
  tf_lite_model = converter.convert()
  # save the converted model 
  open(export_path, 'wb').write(tf_lite_model)

  size = Path(export_path).stat().st_size
  print ("TF model exported to {} size - {} bytes".format(export_path, size))
  return export_path

def onnx2keras(onnx_path):
    export_path = "model.h5"
    onnx_model = onnx.load(onnx_path)
    k_model = onnx_to_keras(onnx_model, ['input'])
    k_model.save(export_path)
    size = Path(export_path).stat().st_size
    print ("ONNX model exported to {} size - {} bytes".format(export_path, size))
    return export_path

def keras2tflite(keras_path):
  export_path = "model.tflite"
  # Load the tensorflow model
  model = tf.keras.models.load_model(keras_path)

  # TFlite model
  # converter = tf.lite.TFLiteConverter.from_keras_model(model) # TF 2.x
  converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_path) # TF 1.x
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()

  # Save the TF Lite model.
  with tf.io.gfile.GFile(export_path, 'wb') as f:
    f.write(tflite_model)
  size = Path(export_path).stat().st_size
  print ("Keras model exported to {} size - {} bytes".format(export_path, size))

  return export_path

def tflite2cpp(tflite_path):
  export_path = "model.cc"
  os.system('xxd -i '+tflite_path+' > '+export_path)
  size = Path(export_path).stat().st_size
  print ("TFlite model exported to {} size - {} bytes".format(export_path, size))
  return export_path

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
    x = x.view(-1, 8*8*32)
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

images = images[0].reshape(1, 2, 128, 128)

model = SiameseNet(1, 1)
model.load_state_dict(torch.load('model.pth'))


def printTensors(pb_file):

    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name)


# printTensors("model.pb")

onnx_path = export2onnx(images, label, model)
tf_path = onnx2tf(onnx_path)
tflite_path = tf2tflite(tf_path)
cpp_path = tflite2cpp(tflite_path)