from torchvision import transforms
import torch
import numpy as np
import cv2
from PIL import Image

unloader = transforms.ToPILImage()
loader = transforms.Compose([
    transforms.ToTensor()])

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image[0]

    image = unloader(image)

    return image


def usm(image_path):  # usm锐化操作（提取轮廓）
    src = cv2.imread(image_path)

    gaussian = cv2.GaussianBlur(src, (7, 7), 7)
    dst2 = cv2.addWeighted(src, 2, gaussian, -1, 0)  # 2*src-gaussian*(-1)
    # print(np.shape(dst2))

    return dst2


if __name__ == '__main__':

    # a = torch.randn(100, 3, 50, 50)
    # b = tensor_to_PIL(a)
    # print(np.shape(b))  # (50, 50, 3)
    dst2 = usm("1.jpg")
    cv2.imshow("dst2", dst2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



