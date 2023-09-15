from PIL import Image
import numpy as np
image1 =Image.open(r'G:\project\mmsegmentation\data\Taiyuan_city\ann_dir\train\crop_1359.png')
images1 = np.asarray(image1)#转化成数组以后，iamges中存储的是图片的像素值。
print(images1)

image2 =Image.open(r'G:\数据集\Taiyuan\train\mask_512\crop_44.png')
images2 = np.asarray(image2)#转化成数组以后，iamges中存储的是图片的像素值。
print(images2)