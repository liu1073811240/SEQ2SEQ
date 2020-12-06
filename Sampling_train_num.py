import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
from utils2 import usm
import cv2

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lower_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']
upper_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']


class Sampling(data.Dataset):
    def __init__(self, root):
        self.transform = data_transforms
        self.imgs = []
        self.labels = []
        for filenames in os.listdir(root):
            x = os.path.join(root, filenames)
            y = filenames.split('.')[0]
            self.imgs.append(x)
            self.labels.append(y)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = usm(img_path)  # usm锐化操作（提取图片轮廓）

        # img = Image.open(img_path)
        # img = self.transform(img)

        img = self.transform(img)
        label = self.labels[index]
        # print(label)  # f3iX

        label = self.StrtoLabel(label)  # 将字母转成数字表示，方便做one-hot
        label = self.one_hot(label)
        # print(label)

        return img, label

    def one_hot(self, x):

        z = np.zeros(shape=[4, 62])
        for i in range(4):
            index = int(x[i])
            z[i][index] = 1

        return z

    def StrtoLabel(self, Str):

        label = []
        for i in range(0, 4):
            if Str[i] >= '0' and Str[i] <= '9':
                # print(ord(Str[i]))  # 1通过ascll转为49
                # print(ord('0'))  # 0通过ascll转为48
                label.append(ord(Str[i]) - ord('0'))
            elif Str[i] >= 'a' and Str[i] <= 'z':

                label.append(ord(Str[i]) - ord('a') + 10)
            else:
                label.append(ord(Str[i]) - ord('A') + 36)
        return label


if __name__ == '__main__':
    samping = Sampling("./code")
    dataloader = data.DataLoader(samping, 10, shuffle=True)
    for i, (img, label) in enumerate(dataloader):
        print(i)
        print(img.shape)
        print(label.shape)
