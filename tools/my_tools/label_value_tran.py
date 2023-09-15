import os
from tqdm import tqdm
import numpy as np
import cv2
import os
from PIL import Image



def create_txt(paths,txt_path):
    with open(txt_path,'a') as f :
        filenames=os.listdir(paths)
        for filename in filenames:
            if os.path.splitext(filename)[1]=='.png':
                out_path=paths + '\\' + filename
                #print(out_path)
                #f.writable(out_path+'\n')
                f.write(out_path+'\n')
def transform_value(txt_path,gray_save_path):
    str = 'runing'
    with open(txt_path, 'r') as f:
        file_names = f.readlines()
        for name in tqdm(file_names):
            name = name.strip('\n')  # 去掉换行符
            dir_name, name = os.path.split(name)
            label_name = name
            label_url = os.path.join(dir_name, label_name)
            mask = cv2.imread(label_url,cv2.IMREAD_GRAYSCALE)
            mask = mask.astype('uint8')
            #mask = np.array([0,1,2,3,4,5,6,7,8,9])
            label_mask = mask + 1
            label_mask = np.where(label_mask ==10,0,label_mask)
            label_mask = Image.fromarray(label_mask)
            label_mask = label_mask.convert('L')
            os.makedirs(gray_save_path, exist_ok=True)
            label_mask.save(gray_save_path + label_name)
    os.remove(txt_path)


if __name__ == '__main__':
    gray_paths = r'G:\masks_512'
    txt_path = r'G:\mask.txt'
    gray_save_path = 'G:/masks/'

    create_txt(gray_paths,txt_path)
    transform_value(txt_path,gray_save_path)