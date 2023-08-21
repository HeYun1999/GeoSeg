import os
import cv2
import csv
import math
import random
import shutil


def copyimg(filepath, testfilepath,trainfilepath,rate):
    #rate为测试集所占份额
    # 从数据集中随机选取20%拷贝到另一文件夹下作为测试集，剩下的80%作为训练集
    # 判断文件夹路径是否存在，如果不存在，则创建，此处是创建多级目录
    if not os.path.isdir(testfilepath):
        os.makedirs(testfilepath)
    if not os.path.isdir(trainfilepath):
        os.makedirs(trainfilepath)
    trainlist= []
    filedir = os.listdir(filepath)
    trainlist = filedir
    ranfilepath = random.sample(filedir, int(rate * len(filedir)))
    print(ranfilepath)
    for filename in ranfilepath:
        child = os.path.join(filepath, filename)
        dest = os.path.join(testfilepath, filename)
        shutil.copy(child, dest)
        trainlist.remove(filename)
        #os.remove(child)
    for filename in trainlist:
        child = os.path.join(filepath, filename)
        dest = os.path.join(trainfilepath, filename)
        shutil.copy(child, dest)


def copy_label(labelpath,testimgpath,trainimgpath,testlabelpath,trainlabelpath):
    # 判断文件夹路径是否存在，如果不存在，则创建，此处是创建多级目录
    if not os.path.isdir(testlabelpath):
        os.makedirs(testlabelpath)
    if not os.path.isdir(trainlabelpath):
        os.makedirs(trainlabelpath)
    # 根据不同文件夹下的图片移动相应图片的标签
    imgdir = os.listdir(testimgpath)
    for img in imgdir:
        labels = os.listdir(labelpath)
        for label in labels:
            if os.path.splitext(label)[0] == os.path.splitext(img)[0]:#
                print('###')
                label_path = os.path.join(labelpath, label)
                test_path = os.path.join(testlabelpath, label)
                print(label_path)
                if labelpath:
                    shutil.copy(label_path, test_path)
                    #os.remove(label_path)

    #train label copy
    imgdir = os.listdir(trainimgpath)
    for img in imgdir:
        labels = os.listdir(labelpath)
        for label in labels:
            if os.path.splitext(label)[0] == os.path.splitext(img)[0]:#
                print('###')
                label_path = os.path.join(labelpath, label)
                train_path = os.path.join(trainlabelpath, label)
                print(label_path)
                if labelpath:
                    shutil.copy(label_path, train_path)
                    #os.remove(label_path)

def copy_label(labelpath,testimgpath,trainimgpath,testlabelpath,trainlabelpath):
    # 判断文件夹路径是否存在，如果不存在，则创建，此处是创建多级目录
    if not os.path.isdir(testlabelpath):
        os.makedirs(testlabelpath)
    if not os.path.isdir(trainlabelpath):
        os.makedirs(trainlabelpath)
    # 根据不同文件夹下的图片移动相应图片的标签
    imgdir = os.listdir(testimgpath)
    for img in imgdir:
        labels = os.listdir(labelpath)
        for label in labels:
            if os.path.splitext(label)[0] == os.path.splitext(img)[0]:#
                print('###')
                label_path = os.path.join(labelpath, label)
                test_path = os.path.join(testlabelpath, label)
                print(label_path)
                if labelpath:
                    shutil.copy(label_path, test_path)
                    #os.remove(label_path)

    #train label copy
    imgdir = os.listdir(trainimgpath)
    for img in imgdir:
        labels = os.listdir(labelpath)
        for label in labels:
            if os.path.splitext(label)[0] == os.path.splitext(img)[0]:#
                print('###')
                label_path = os.path.join(labelpath, label)
                train_path = os.path.join(trainlabelpath, label)
                print(label_path)
                if labelpath:
                    shutil.copy(label_path, train_path)
                    #os.remove(label_path)





if __name__ == "__main__":
    #将rate % 的原图移动到test文件夹中，那么1 - rate% 的原图就是train
    #参数1：最初原图的路径 参数2 测试图像路径 参数3 训练图像路径 参数4 分割比率
    copyimg('../data/vaihingen/images', '../data/vaihingen/test_images', '../data/vaihingen/train_images',0.2)
    #寻找 与已经移动到test集中原图名字相同的label文件，并将其移动到test集中的label路径中
    #参数1 test集中原图的路径 参数2 最初label的路径 参数3 test集中label的路径
    copy_label('../data/vaihingen/labels','../data/vaihingen/test_images','../data/vaihingen/train_images','../data/vaihingen/test_masks','../data/vaihingen/train_masks')
