import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
from PIL import Image
import model
import random
import cv2
import csv
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.model_baseline().to(device)
model = torch.nn.DataParallel(model, device_ids=[0,1])

model.load_state_dict(torch.load('./model_final' + '.pb'))

transforms_valid = transforms.Compose([
        transforms.CenterCrop(600),
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

data_test_dir = '/data/data_weather/test/'
ret_submit_dir = './ret_submit.csv'

test_file_list = os.listdir(data_test_dir)
ret_submit_write = csv.writer(open(ret_submit_dir, 'a', newline=''), dialect='excel')

model = model.eval()
for i in range(len(test_file_list)):
    temp_result = []
    temp_dir = data_test_dir + test_file_list[i]
    
    try:
        img_temp_RGB = Image.open(temp_dir)
        if img_temp_RGB.mode != 'RGB':
            img_temp_RGB = img_temp_RGB.convert('RGB')
        img_tensor = transforms_valid(img_temp_RGB)
        img_tensor = img_tensor[:3,:,:]
        img_tensor = torch.unsqueeze(img_tensor, 0)
        img_tensor = img_tensor.to(device)
        outputs = model(img_tensor)
        _, predicts = torch.max(outputs.data, 1)
    

        f_process = float(i+1)/float(len(test_file_list)) * 100 
        print('{:.2f} %'.format(f_process), end='\r')

        label_predict = int(predicts.item()) + 1
    except OSError as e:
        print(e)
        print(img_tensor.shape)
        print('有问题！')
        label_predict = 0
        continue
    
    file_name = test_file_list[i]
    temp_result.append(file_name)
    temp_result.append(label_predict)
    ret_submit_write.writerow(temp_result)


