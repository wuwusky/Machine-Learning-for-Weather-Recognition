import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
from torch.autograd import Variable
from PIL import Image
import model
import random
# import cv2
import csv
from pre_models import resnext
from pre_models import senet
import torch.nn.functional as F
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class my_dataset(Dataset):
    def __init__(self, list_data, transforms=None):
        self.list_data = list_data
        self.transforms = transforms
    def __getitem__(self, item):
        label_temp_1 = self.list_data[item][1]
        temp_dir = '/data/data_weather/train/'+ self.list_data[item][0]
        img_temp_RGB = Image.open(temp_dir)
        if img_temp_RGB.mode != 'RGB':
            img_temp_RGB = img_temp_RGB.convert('RGB')
        if transforms != None:
            img_tensor = self.transforms(img_temp_RGB)
        
        label_tensor_1 = torch.from_numpy(np.array(int(label_temp_1)-1, dtype=np.float32))

        return img_tensor, label_tensor_1.long()
    def __len__(self):
        return len(self.list_data)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1-lam)*x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_mixup = lam * criterion(pred, y_a) + (1-lam)*criterion(pred, y_b)
    return loss_mixup

def loss_fun(outputs, labels):
    list_label_type = [0]*4
    list_weight = []
    for i in range(outputs.size(0)):
        list_label_type[labels[i]] += 1
    for i in range(len(list_label_type)):
        loss_prob = 32.0 / float(list_label_type[i])
        list_weight.append(loss_prob)
    tensor_weight = torch.from_numpy(np.array(list_weight, dtype=np.float32))
    return tensor_weight

class FocalLoss(nn.Module):
    
    def __init__(self, device, gamma = 2.0):
        super(FocalLoss, self).__init__()
        self.device = device
        self.gamma = torch.tensor(gamma, dtype = torch.float32).to(device)
        self.eps = 1e-6
        
        # self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none').to(device)
        
    def forward(self, input, target):
        
        BCE_loss = F.cross_entropy(input, target, reduction='none').to(self.device)
        # BCE_loss = self.BCE_loss(input, target)
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss =  (1-pt)**self.gamma * BCE_loss
        
        return F_loss.mean() 


if __name__ == '__main__':
    batch_size = 4
    n_workers = 8
    learn_rate = 1e-3
    train_dir = './Train_label.csv'
    data_dir = '/data/data_weather/'
    flag_mixup = False
    flag_pretrain = False
    input_size = 512
    crop_sieze = 800
    iter_num_multiple = 1.
    flag_need_from_zero = False

    max_epoch = int(50. * iter_num_multiple)

    list_rate_train = []
    list_rate_val = []


    model = senet.senet154().to(device)
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    if flag_pretrain:
        model.load_state_dict(torch.load('./model_final' + '.pb'))
    # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=1e-5)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learn_rate, weight_decay=0.)

    transforms_train = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((crop_sieze, crop_sieze), pad_if_needed=True),
        # transforms.CenterCrop((crop_sieze, crop_sieze)),
        transforms.Resize((input_size, input_size)),
        transforms.ColorJitter(0.25, 0.25),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    transforms_valid = transforms.Compose([
        transforms.CenterCrop((crop_sieze, crop_sieze)),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    file_reader = csv.reader(open(train_dir, encoding='utf-8'))
    list_data = []
    list_data_valid = []
    list_data_train = []

    criterion = nn.CrossEntropyLoss()
    list_data_all = []
    for i, temp_info in enumerate(file_reader):
        if i > 0:
            if (i+1) % 50 == 0:
                list_data_valid.append(temp_info)
            else:
                list_data_train.append(temp_info)
            list_data_all.append(temp_info)

    dataset_train = my_dataset(list_data_train, transforms=transforms_train)
    dataset_valid = my_dataset(list_data_valid, transforms=transforms_valid)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, num_workers=n_workers, drop_last=True)
    current_lr = learn_rate
    max_rate_val = 0
    flag_epoch = 0
    for epoch in range(max_epoch):
        time_start = time.time()
        model = model.train()
        total = 0
        num_correct = 0
        for i ,(imgs, labels) in enumerate(loader_train):
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels = labels.squeeze_()
            optimizer.zero_grad()

            if flag_mixup:
                outputs_for_predict = model(imgs)
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, alpha=0.5)
                imgs, labels_a, labels_b = map(Variable, (imgs, labels_a, labels_b))

                outputs, outputs_aux = model(imgs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                loss_aux = mixup_criterion(criterion, outputs_aux, labels_a, labels_b, lam)
                loss = loss + loss_aux
                outputs = outputs_for_predict
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                # loss_aux = criterion(outputs_aux, labels)
                # loss = loss + loss_aux*0.5
            _, predicts = torch.max(outputs.data, 1)
            total += labels.size(0)
            num_correct += (predicts==labels).sum().item()
            if (i+1) % 8 == 0:
                print('Epoch:[{}/{}], Step:[{}/{}], loss:{:.6f}, lr:{:.6f}'.format(epoch+1, max_epoch, i+1, len(loader_train), loss.item(), current_lr))
            loss.backward()
            optimizer.step()
        acc_train = 100.0 * float(num_correct)/float(total)
        list_rate_train.append(acc_train)
        
        model = model.eval()
        total = 0
        num_correct = 0
        for (imgs, labels) in loader_valid:
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels = labels.squeeze_()
            outputs = model(imgs)
            _, predicts = torch.max(outputs.data, 1)
            total += labels.size(0)
            num_correct += (predicts==labels).sum().item()
        acc = 100.0 * float(num_correct)/float(total)
        list_rate_val.append(acc)
        if acc >= max_rate_val:
            max_rate_val = acc
            torch.save(model.state_dict(), './model_best' + '.pb' )
            flag_epoch = epoch + 1
        time_cost = time.time() - time_start
        print('Epoch:{}, Val_rate:{:.2f}%, train_rate:{:.2f}%, time_cost:{:.2f}s'.format(epoch+1, acc, acc_train, time_cost))

        
        if flag_need_from_zero:
            if epoch+1 == int(15 * iter_num_multiple) or epoch+1 == int(25 * iter_num_multiple) or epoch+1 == int(35 * iter_num_multiple) or epoch+1 == int(40 * iter_num_multiple) or epoch+1 == int(45 * iter_num_multiple):
                current_lr = current_lr * 0.1
                update_lr(optimizer, current_lr)
        else:
            if epoch+1 == int(20 * iter_num_multiple) or epoch+1 == int(35 * iter_num_multiple) or epoch+1 == int(45 * iter_num_multiple):
                current_lr = current_lr * 0.1
                update_lr(optimizer, current_lr)

    model = model.eval()
    torch.save(model.state_dict(), './model_final' + '.pb' )
    print('The most best rate_val:{:.2f}% , the best iter epoch is:{}'.format(max_rate_val, flag_epoch))
    log_train = open('./log.txt', 'w')
    for i in range(len(list_rate_train)):
        temp_line = str(i+1) + '  ' + str(round(list_rate_train[i], 2)) + '%' + '  ' + str(round(list_rate_val[i], 2)) + '%' + '\n'
        log_train.write(temp_line)
    log_train.close()
