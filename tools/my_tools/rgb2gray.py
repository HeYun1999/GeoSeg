from tqdm import tqdm
import numpy as np
import cv2
import os
from PIL import Image

# 标签中每个RGB颜色的值
VOC_COLORMAP = np.array([[250,250,250],[0, 255, 0], [34, 139, 34], [107, 142, 35], [0, 0, 255],
                         [255, 0, 0], [192, 192, 192],[128, 42, 42],[254, 252, 193],[255, 255, 255]])
# 标签其标注的类别
VOC_CLASSES = ['background','farmland', 'woodland', 'grassland', 'waters', 'building',
               'Hardened_surface', 'Heap_digging','road','others']

# 处理txt中的对应图像
txt_path = r'G:\mask.txt'
# 标签所在的文件夹
label_file_path = r'G:\masks_512'
# 处理后的标签保存的地址
gray_save_path = 'G:/masks/'

with open(txt_path, 'r') as f:
    file_names = f.readlines()
    for name in tqdm(file_names):
        name = name.strip('\n')  # 去掉换行符
        dir_name, name = os.path.split(name)
        label_name = name
        label_url = os.path.join(label_file_path, label_name)
        mask = cv2.imread(label_url)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # 通道转换
        mask = mask.astype('uint8')
        #制作一个全为9的，尺度与源标签大小一样的tensor,标签只有0~8，9表明是没分割的区域，例如卫星图片中全黑区域
        label_mask = np.full((mask.shape[0], mask.shape[1]), 0,dtype=np.uint8)
        #label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        # 标签处理
        #ii保存索引，label保存像素值例[0, 255, 0]
        for ii, label in enumerate(VOC_COLORMAP):
            #all()函数用于判断整个数组中的元素的值是否全部满足条件，
            #如果满足条件返回True，否则返回False。本质上讲，all()实现了或(AND)运算
            locations = np.all(mask == label, axis=-1)
            #如果对应像素值mask == label，则将此像素值标为标签对应的索引值
            label_mask[locations] = ii
        # 标签保存
        label_mask = Image.fromarray(label_mask)
        label_mask.save(gray_save_path + label_name)
        #cv2.imwrite(gray_save_path + label_name, label_mask)