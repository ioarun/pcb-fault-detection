import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import random
import torch

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

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print (input_details[0])
print (output_details[0])
# # Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)