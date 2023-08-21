import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
import random

SEED = 42

#opencv 默认图像基准为BGR，因此对图片进行转换时，要先转换成BGR，即RGB2BGR
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


ImSurf = np.array([255, 255, 255])  # label 0
Building = np.array([255, 0, 0]) # label 1
LowVeg = np.array([255, 255, 0]) # label 2
Tree = np.array([0, 255, 0]) # label 3
Car = np.array([0, 255, 255]) # label 4
Clutter = np.array([0, 0, 255]) # label 5
Boundary = np.array([0, 0, 0]) # label 6
num_classes = 6


# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="../data/vaihingen/train_images")
    parser.add_argument("--mask-dir", default="../data/vaihingen/train_masks")
    parser.add_argument("--output-img-dir", default="../data/vaihingen/train/images")
    parser.add_argument("--output-mask-dir", default="../data/vaihingen/train/masks")
    parser.add_argument("--eroded", action='store_true')
    parser.add_argument("--gt", action='store_true')
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--val-scale", type=float, default=1.0)
    parser.add_argument("--split-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    return parser.parse_args()

#分割时，如果尺寸无法整分，则需要填充一部分，输出RGB格式的填充后图片
def get_img_mask_padded(image, mask, patch_size, mode):
    img, mask = np.array(image), np.array(mask)
    oh, ow = img.shape[0], img.shape[1]#图片长和宽
    rh, rw = oh % patch_size, ow % patch_size
    #如果图片长和宽能被要分割的尺寸整除的话，则不填充，否者填充差值
    width_pad = 0 if rw == 0 else patch_size - rw
    height_pad = 0 if rh == 0 else patch_size - rh
    #新的图片长宽，应该是原尺寸加上需要填充的部分
    h, w = oh + height_pad, ow + width_pad
    #对于图片与标签进行填充，填充值为关于右，下侧边界的对称值
    pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', border_mode=cv2.BORDER_CONSTANT, value=0)(image=img)
    pad_mask = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', border_mode=cv2.BORDER_CONSTANT, value=6)(image=mask)
    img_pad, mask_pad = pad_img['image'], pad_mask['image']
    #将图片转成rgb图像
    img_pad = cv2.cvtColor(np.array(img_pad), cv2.COLOR_RGB2BGR)
    mask_pad = cv2.cvtColor(np.array(mask_pad), cv2.COLOR_RGB2BGR)
    return img_pad, mask_pad


def pv2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def car_color_replace(mask):
    mask = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)#颜色空间转换，从rgb转成bgr
    mask[np.all(mask == [0, 255, 255], axis=-1)] = [0, 204, 255]#将mask中所有像素值为0 255 255的转换成0 204 255
    #返回mask
    return mask

#将RGB的3通道标签转换成单通道的灰度图
def rgb_to_2D_label(_label):
    _label = _label.transpose(2, 0, 1)#换通道
    label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)#以原图片尺寸，制成一个新的h x w的2D图片用来存储，新的标签值
    label_seg[np.all(_label.transpose([1, 2, 0]) == ImSurf, axis=-1)] = 0
    label_seg[np.all(_label.transpose([1, 2, 0]) == Building, axis=-1)] = 1
    label_seg[np.all(_label.transpose([1, 2, 0]) == LowVeg, axis=-1)] = 2
    label_seg[np.all(_label.transpose([1, 2, 0]) == Tree, axis=-1)] = 3
    label_seg[np.all(_label.transpose([1, 2, 0]) == Car, axis=-1)] = 4
    label_seg[np.all(_label.transpose([1, 2, 0]) == Clutter, axis=-1)] = 5
    label_seg[np.all(_label.transpose([1, 2, 0]) == Boundary, axis=-1)] = 6
    return label_seg
#对图片进行变换，并将结果放入一个列表中
#图像列表：包含了原图、水平反转后的图、垂直反转后的图、填充后的图
#标签列表：包含了原图、水平反转后的图、垂直反转后的图、填充后的且转换成灰度图的标签
def image_augment(image, mask, patch_size, mode='train', val_scale=1.0):
    image_list = []
    mask_list = []
    image_width, image_height = image.size[1], image.size[0]#获得输入图像尺寸
    mask_width, mask_height = mask.size[1], mask.size[0]

    assert image_height == mask_height and image_width == mask_width#图片与标签尺寸对应了才不报错
    if mode == 'train':
        # resize_0 = Resize(size=(int(image_width * 0.25), int(image_height * 0.25)))
        # resize_1 = Resize(size=(int(image_width * 0.5), int(image_height * 0.5)))
        # resize_2 = Resize(size=(int(image_width * 0.75), int(image_height * 0.75)))
        # resize_3 = Resize(size=(int(image_width * 1.25), int(image_height * 1.25)))
        # resize_4 = Resize(size=(int(image_width * 1.5), int(image_height * 1.5)))
        # resize_5 = Resize(size=(int(image_width * 1.75), int(image_height * 1.75)))
        # resize_6 = Resize(size=(int(image_width * 2.0), int(image_height * 2.0)))
        # image_resize_0, mask_resize_0 = resize_0(image.copy()), resize_0(mask.copy())
        # image_resize_1, mask_resize_1 = resize_1(image.copy()), resize_1(mask.copy())
        # image_resize_2, mask_resize_2 = resize_2(image.copy()), resize_2(mask.copy())
        # image_resize_3, mask_resize_3 = resize_3(image.copy()), resize_3(mask.copy())
        # image_resize_4, mask_resize_4 = resize_4(image.copy()), resize_4(mask.copy())
        # image_resize_5, mask_resize_5 = resize_5(image.copy()), resize_5(mask.copy())
        # image_resize_6, mask_resize_6 = resize_6(image.copy()), resize_6(mask.copy())
        h_vlip = RandomHorizontalFlip(p=1.0)#对图片进行随机水平反转，p=1表示100%的水平翻转
        v_vlip = RandomVerticalFlip(p=1.0)#对图片进行随机垂直反转，p=1表示100%的垂直翻转
        # crop_1 = RandomCrop(size=(int(image_width*0.75), int(image_height*0.75)))
        # crop_2 = RandomCrop(size=(int(image_width * 0.5), int(image_height * 0.5)))
        # color = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        image_h_vlip, mask_h_vlip = h_vlip(image.copy()), h_vlip(mask.copy())#水平反转
        image_v_vlip, mask_v_vlip = v_vlip(image.copy()), v_vlip(mask.copy())#垂直反转
        # image_crop_1, mask_crop_1 = crop_1(image.copy()), crop_1(mask.copy())
        # image_crop_2, mask_crop_2 = crop_2(image.copy()), crop_2(mask.copy())
        # image_color = color(image.copy())

        image_list_train = [image, image_h_vlip, image_v_vlip]#原图、水平翻转后、垂直反转后，三张图片制成一个列表存放
        mask_list_train = [mask, mask_h_vlip, mask_v_vlip]
        # image_list_train = [image]
        # mask_list_train = [mask]
        for i in range(len(image_list_train)):#依次取出这三张图片
            image_tmp, mask_tmp = get_img_mask_padded(image_list_train[i], mask_list_train[i], patch_size, mode)#填充图片，防止分割时无法整除
            mask_tmp = rgb_to_2D_label(mask_tmp.copy())#将rgb的3通道标签，转换成单通道的灰度图标签
            image_list.append(image_tmp)#将填充好的图片，以及转换好的灰度图标签加入列表中
            mask_list.append(mask_tmp)
    else:#如果时测试，或验证 模式的话，则进行
        rescale = Resize(size=(int(image_width * val_scale), int(image_height * val_scale)))
        image, mask = rescale(image.copy()), rescale(mask.copy())
        image, mask = get_img_mask_padded(image.copy(), mask.copy(), patch_size, mode)
        mask = rgb_to_2D_label(mask.copy())
        image_list.append(image)
        mask_list.append(mask)
    return image_list, mask_list


def randomsizedcrop(image, mask):
    # assert image.shape[:2] == mask.shape
    h, w = image.shape[0], image.shape[1]
    crop = albu.RandomSizedCrop(min_max_height=(int(3*h//8), int(h//2)), width=h, height=w)(image=image.copy(), mask=mask.copy())
    img_crop, mask_crop = crop['image'], crop['mask']
    return img_crop, mask_crop

#对图像进行增强
#制成一个列表，包含原始图，水平反转，垂直反转、90度旋转
def car_aug(image, mask):
    assert image.shape[:2] == mask.shape
    v_flip = albu.VerticalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    h_flip = albu.HorizontalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    rotate_90 = albu.RandomRotate90(p=1.0)(image=image.copy(), mask=mask.copy())
    # blur = albu.GaussianBlur(p=1.0)(image=image.copy())
    image_vflip, mask_vflip = v_flip['image'], v_flip['mask']
    image_hflip, mask_hflip = h_flip['image'], h_flip['mask']
    image_rotate, mask_rotate = rotate_90['image'], rotate_90['mask']
    # blur_image = blur['image']
    image_list = [image, image_vflip, image_hflip, image_rotate]
    mask_list = [mask, mask_vflip, mask_hflip, mask_rotate]

    return image_list, mask_list


def vaihingen_format(inp):#数据集格式化
    (img_path, mask_path, imgs_output_dir, masks_output_dir, eroded, gt, mode, val_scale, split_size, stride) = inp
    img_filename = os.path.splitext(os.path.basename(img_path))[0]
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    if eroded:
        mask_path = mask_path[:-4] + '_noBoundary.tif'
    img = Image.open(img_path).convert('RGB')#将图片转换成rgb图
    mask = Image.open(mask_path).convert('RGB')
    if gt:
        mask_ = car_color_replace(mask)#将rgb图转换成bgr图，在将一个像素值，用另一个像素值代替
        out_origin_mask_path = os.path.join(masks_output_dir + '/origin/', "{}.tif".format(mask_filename))#新的输出路径
        cv2.imwrite(out_origin_mask_path, mask_)#把修改好的mask放入路径下
    # print(img_path)
    # print(img.size, mask.size)
    # img and mask shape: WxHxC
    # 图像列表：包含了原图、水平反转后的图、垂直反转后的图、填充后的图
    # 标签列表：包含了原图、水平反转后的图、垂直反转后的图、填充后的且转换成灰度图的标签
    image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(), patch_size=split_size,
                                          mode=mode, val_scale=val_scale)
    assert img_filename == mask_filename and len(image_list) == len(mask_list)
    for m in range(len(image_list)):
        k = 0
        img = image_list[m]
        mask = mask_list[m]
        assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]
        if gt:
            mask = pv2rgb(mask)#如果判断为真，则灰度图标签再转成rgb标签
        #两层循环，对图片进行从左到右，从上到下的分割
        for y in range(0, img.shape[0], stride):
            for x in range(0, img.shape[1], stride):
                img_tile = img[y:y + split_size, x:x + split_size]
                mask_tile = mask[y:y + split_size, x:x + split_size]
                #\为续行符
                #判断如果分割图片尺寸正常，则继续
                if img_tile.shape[0] == split_size and img_tile.shape[1] == split_size \
                        and mask_tile.shape[0] == split_size and mask_tile.shape[1] == split_size:
                    image_crop, mask_crop = randomsizedcrop(img_tile, mask_tile)#对切割好的图片，进行随机裁剪
                    bins = np.array(range(num_classes + 1))
                    class_pixel_counts, _ = np.histogram(mask_crop, bins=bins)#统计图片像素值为0 - 7的个数，虽然时六类，但这个函数指定范围为左闭右开的区间，因此要+1才能统计所有类别
                    cf = class_pixel_counts / (mask_crop.shape[0] * mask_crop.shape[1])
                    #图像分割后的保存工作
                    if cf[4] > 0.1 and mode == 'train':#如果满足条件，则再进行数据增强
                        car_imgs, car_masks = car_aug(image_crop, mask_crop)
                        for i in range(len(car_imgs)):
                            out_img_path = os.path.join(imgs_output_dir,
                                                        "{}_{}_{}_{}.tif".format(img_filename, m, k, i))
                            cv2.imwrite(out_img_path, car_imgs[i])

                            out_mask_path = os.path.join(masks_output_dir,
                                                         "{}_{}_{}_{}.png".format(mask_filename, m, k, i))
                            cv2.imwrite(out_mask_path, car_masks[i])
                    else:
                        out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}.tif".format(img_filename, m, k))
                        cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(masks_output_dir, "{}_{}_{}.png".format(mask_filename, m, k))
                        cv2.imwrite(out_mask_path, mask_tile)

                k += 1


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    imgs_dir = args.img_dir
    masks_dir = args.mask_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    gt = args.gt
    eroded = args.eroded
    mode = args.mode
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride
    img_paths = glob.glob(os.path.join(imgs_dir, "*.tif"))#读取图片
    mask_paths_raw = glob.glob(os.path.join(masks_dir, "*.tif"))#读取label
    if eroded:
        mask_paths = [(p[:-15] + '.tif') for p in mask_paths_raw]
    else:
        mask_paths = mask_paths_raw
    img_paths.sort()#排序，默认降序排列
    mask_paths.sort()

    if not os.path.exists(imgs_output_dir):#检测输出路径，若不存在则创建
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        if gt:
            os.makedirs(masks_output_dir+'/origin')

    inp = [(img_path, mask_path, imgs_output_dir, masks_output_dir, eroded, gt, mode, val_scale, split_size, stride)
           for img_path, mask_path in zip(img_paths, mask_paths)]#zip将对应索引的值，打包成元组，在转成列表输出。说白了就是构造img_path与mask_path要一一对应

    t0 = time.time()
    # mpp.Pool（）多线程进行后续操作，
    mpp.Pool(processes=mp.cpu_count()).map(vaihingen_format, inp)#。map函数是一个映射，就是将inp值依次放入vaihingen_format函数中
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))


