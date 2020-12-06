import os
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


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
        img = Image.open(img_path)
        img = self.transform(img)
        label = self.labels[index]
        # print(label)
        label = self.one_hot(label)
        # print(label)


        return img, label

    def one_hot(self, x):
        z = np.zeros(shape=[4, 10])
        for i in range(4):
            index = int(x[i])
            z[i][index] = 1
        return z


if __name__ == '__main__':
    samping = Sampling("./code2")
    dataloader = data.DataLoader(samping, 10,
                                 shuffle=True)
    for i, (img, label) in enumerate(dataloader):
        print(i)
        print(img.shape)
        print(label.shape)
