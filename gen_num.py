from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import os
import numpy as np
from utils2 import usm
import cv2


# 随机字母
def ranNum():
    a = str(random.randint(0, 9))
    # print(a)  # 0-9

    a = chr(random.randint(48, 57))
    # print(a)  # 0~9

    b = chr(random.randint(65, 90))  # 大写字母
    # print(b)  # A~Z

    c = chr(random.randint(97, 122))  # 小写字母
    # print(c)  # a~z

    d = ord(c)
    # print(d)  # 97~122

    x = random.choice([a, b, c])
    # print(x)

    return x


# 随机颜色
def ranColor():
    return (random.randint(0, 128),
            random.randint(0, 128),
            random.randint(0, 128))


# 240*60
w = 240
h = 60
# 创建字体对象
font = ImageFont.truetype("arial.ttf", 40)
for i in range(10240):
    # 随机生成像素矩阵
    img_arr = np.random.randint(128, 512, (h, w, 3))
    # print(img_arr)
    # print(np.shape(img_arr))  # (60, 240, 3)
    # 将像素矩阵转换成背景图片
    image = Image.fromarray(np.uint8(img_arr))

    # 创建Draw对象
    draw = ImageDraw.Draw(image)
    # 填充文字像素颜色
    filename = ""
    for j in range(4):
        ch = ranNum()
        # print(ch)  # m
        filename += ch
        # print(filename)  # m

        # 给字体之间增加间隔
        draw.text((40 * j + 20 * (j + 1), 10), ch, font=font, fill=ranColor())

    print(filename)  # TykA

    # 模糊:
    image = image.filter(ImageFilter.BLUR)

    # image.show()
    if not os.path.exists("./code"):
        os.makedirs("./code")
    image_path = r"./code"
    image.save("{0}/{1}.jpg".format(image_path, filename))
    print(i)
