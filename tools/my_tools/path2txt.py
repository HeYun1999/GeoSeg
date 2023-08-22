import os
paths='E:/MYDATASET/512RGB/mask'
f=open('E:/MYDATASET/512RGB/mask.txt','r+')
filenames=os.listdir(paths)
for filename in filenames:
    if os.path.splitext(filename)[1]=='.png':
        out_path="E:/MYDATASET/512RGB/mask/"+filename
        print(out_path)
        #f.writable(out_path+'\n')
        f.write(out_path+'\n')
f.close()