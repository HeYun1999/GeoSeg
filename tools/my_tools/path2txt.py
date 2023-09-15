import os
paths=r'G:\mask_512'
f=open(r'G:\mask.txt','r+')
filenames=os.listdir(paths)
for filename in filenames:
    if os.path.splitext(filename)[1]=='.png':
        out_path=paths + '\\' + filename
        print(out_path)
        #f.writable(out_path+'\n')
        f.write(out_path+'\n')
f.close()