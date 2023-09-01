import os
paths=r'G:\project\Geoseg\data\taiyuan\test\masks_rgb'
f=open(r'G:\project\Geoseg\data\taiyuan\test\mask.txt','r+')
filenames=os.listdir(paths)
for filename in filenames:
    if os.path.splitext(filename)[1]=='.png':
        out_path="G:\\project\\Geoseg\\data\\taiyuan\\test\\mask\\"+filename
        print(out_path)
        #f.writable(out_path+'\n')
        f.write(out_path+'\n')
f.close()