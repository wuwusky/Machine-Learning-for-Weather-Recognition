import csv
from PIL import Image
import torchvision.transforms as transforms
import os

train_dir = './Train_label.csv'
data_root = '/data/data_weather/train/'
file_reader = csv.reader(open(train_dir, encoding='utf-8'))

bad_imgs = 0
for i, file_info in enumerate(file_reader):
    if i > 0:
        file_dir = file_info[0]
        file_label = file_info[1]
        img_RGB = Image.open(data_root + file_dir)
        if img_RGB.mode != 'RGB':
            img_RGB = img_RGB.convert('RGB')
        try:
            temp_transform = transforms.ToTensor()
            img_temp = temp_transform(img_RGB)
        except OSError as e:
            bad_imgs += 1
            print(file_dir)
            continue
print(bad_imgs)
        # print(img_temp.shape)
        # temp_transform = transforms.Resize((256,256)))
        # img_RGB = temp_transform(img_RGB)
        # print(img_RGB)

data_test_dir = '/data/data_weather/test/'
test_file_list = os.listdir(data_test_dir)
bad_imgs = 0
for i in range(len(test_file_list)):
    temp_dir = data_test_dir + test_file_list[i]
    img_RGB = Image.open(temp_dir)
    if img_RGB.mode != 'RGB':
        img_RGB = img_RGB.convert('RGB')
        try:
            temp_transform = transforms.ToTensor()
            img_temp = temp_transform(img_RGB)
        except OSError as e:
            bad_imgs += 1
            print(file_dir)
            continue
print(bad_imgs)