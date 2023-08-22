import os
import random
import shutil





def split_files(img,label, new_img, new_label, split_ratio):
    img_file_list = os.listdir(img)
    random.shuffle(img_file_list)
    split_index = int(len(img_file_list) * split_ratio)
    print(len(img_file_list))
    print(split_index)


    top = img_file_list[:split_index]
    print(len(top))
    under = img_file_list[split_index:]
    print(len(under))

    for file_name in top:
        #读取原图像
        img_path = os.path.join(img, file_name)
        #读取原图像对应的标签
        label_path = os.path.join(label, file_name)
        #制作目标路径
        a = 'train'
        destination_img_path = os.path.join(a, file_name)
        destination_img_path = os.path.join(new_img, destination_img_path)

        destination_label_path = os.path.join(a, file_name)
        destination_label_path = os.path.join(new_label, destination_label_path)

        shutil.copy(img_path, destination_img_path)
        shutil.copy(label_path, destination_label_path)

        #进度条

    for file_name in under:

        # 读取原图像
        img_path = os.path.join(img, file_name)
        # 读取原图像对应的标签
        label_path = os.path.join(label, file_name)
        # 制作目标路径
        b = 'val'
        destination_img_path = os.path.join(b, file_name)
        destination_img_path = os.path.join(new_img, destination_img_path)
        destination_label_path = os.path.join(b, file_name)
        destination_label_path = os.path.join(new_label, destination_label_path)

        shutil.copy(img_path, destination_img_path)
        shutil.copy(label_path, destination_label_path)
        #进度条

img = 'F:/pycharm/project/mmsegmentation/data/Taiyuan_city/img'  # 源文件夹路径
label ='F:/pycharm/project/mmsegmentation/data/Taiyuan_city/new_mask'
new_img = 'F:/pycharm/project/mmsegmentation/data/Taiyuan_city/img_dir'  # 第一个保存文件的文件夹路径
new_label = 'F:/pycharm/project/mmsegmentation/data/Taiyuan_city/ann_dir'  # 第二个保存文件的文件夹路径
split_ratio = 0.8 # 划分比例

split_files(img,label, new_img, new_label, split_ratio)
