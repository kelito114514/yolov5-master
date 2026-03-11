import os
import shutil

import numpy as np

# 获取图像路径
image_path = r".//small_head_helmet/images"
label_path = r".//small_head_helmet/labels"
# 定义两个列表存放所有的图像路径 标签路径
image_name=[]
label_name=[]

for fn in os.listdir(image_path):
    image_full_path = os.path.join(image_path,fn)
    #print(image_full_path)

    ln = fn.replace(".png",".txt") # 字符，改为“.txt”
    label_full_path = os.path.join(label_path,ln)

    image_name.append(image_full_path)
    label_name.append(label_full_path)
# 划分训练集和测试集
val_split = 0.2
num = len(image_name)
val_num = int(num*val_split)
train_num = num - val_num

# 打乱图像和标签
np.random.seed(123)
np.random.shuffle(image_name)
np.random.seed(123)
np.random.shuffle(label_name)
# 训练图像，训练标签
train_image = image_name[:train_num]
train_label = label_name[:train_num]
# 测试图像，测试标签
val_image = image_name[train_num:]
val_label = label_name[train_num:]

"""
yolo_dataset:
    image:
        train
        val
    labels:
        train
        val
"""
train_image_path = r".//yolo_dataset/images/train"
train_label_path = r".//yolo_dataset/labels/train"
val_image_path = r".//yolo_dataset/images/val"
val_label_path = r".//yolo_dataset/labels/val"
# 判断该文件是否存在，不存在则创建一个文件夹
if not os.path.exists(train_label_path):
    os.makedirs(train_label_path)
if not os.path.exists(train_image_path):
    os.makedirs(train_image_path)
if not os.path.exists(val_image_path):
    os.makedirs(val_image_path)
if not os.path.exists(val_label_path):
    os.makedirs(val_label_path)
# 移动文件
for fn,im in zip(train_image,train_label):
    # 获取完整的图像名称和标签名称
    im_name = os.path.basename(im)
    fn_name = os.path.basename(fn)
    # 获取移动后图像。标签的路径
    new_image_full_path = os.path.join(train_image_path,im_name)
    new_label_full_path = os.path.join(train_label_path,fn_name)
    # 将原来的数据复制一份到新的路径
    shutil.copy2(im,new_image_full_path)
    shutil.copy2(fn,new_label_full_path)

for fn,im in zip(val_image,val_label):
    # 获取完整的图像名称和标签名称
    im_name = os.path.basename(im)
    fn_name = os.path.basename(fn)
    # 获取移动后图像。标签的路径
    new_image_full_path = os.path.join(val_image_path,im_name)
    new_label_full_path = os.path.join(val_label_path,fn_name)
    # 将原来的数据复制一份到新的路径
    shutil.copy2(im,new_image_full_path)
    shutil.copy2(fn,new_label_full_path)