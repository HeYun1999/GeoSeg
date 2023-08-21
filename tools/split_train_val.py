import os
import cv2
import csv
import math
import random
import shutil


def move(filepath, testfilepath,rate):
    #rate为测试集所占份额
    # 从数据集中随机选取20%移动到另一文件夹下作为测试集，剩下的80%作为训练集
    # 判断文件夹路径是否存在，如果不存在，则创建，此处是创建多级目录
    if not os.path.isdir(testfilepath):
        os.makedirs(testfilepath)
    filedir = os.listdir(filepath)
    ranfilepath = random.sample(filedir, int(rate * len(filedir)))
    print(ranfilepath)
    for filename in ranfilepath:
        child = os.path.join(filepath, filename)
        dest = os.path.join(testfilepath, filename)
        shutil.copy(child, dest)
        os.remove(child)


def move_label(imgpath, labelpath, testpath):
    # 判断文件夹路径是否存在，如果不存在，则创建，此处是创建多级目录
    if not os.path.isdir(testpath):
        os.makedirs(testpath)
    # 根据不同文件夹下的图片移动相应图片的标签
    imgdir = os.listdir(imgpath)
    for img in imgdir:
        labels = os.listdir(labelpath)
        for label in labels:
            if os.path.splitext(label)[0] == os.path.splitext(img)[0]:#
                print('###')
                label_path = os.path.join(labelpath, label)
                test_path = os.path.join(testpath, label)
                print(label_path)
                if labelpath:
                    shutil.copy(label_path, test_path)
                    os.remove(label_path)



if __name__ == "__main__":
    #将rate % 的原图移动到test文件夹中，那么1 - rate% 的原图就是train
    #参数1：最初原图的路径 参数2：test集中原图的路径
    move('../data/vaihingen/train/images', '../data/vaihingen/test/images', 0.2)
    #寻找 与已经移动到test集中原图名字相同的label文件，并将其移动到test集中的label路径中
    #参数1 test集中原图的路径 参数2 最初label的路径 参数3 test集中label的路径
    move_label('../data/vaihingen/test/images','../data/vaihingen/train/masks','../data/vaihingen/test/masks')
