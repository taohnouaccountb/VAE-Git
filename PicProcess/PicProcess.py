import threading
import os
import random
import numpy as np
from copy import deepcopy
from PIL import Image
THREAD_NUM = 16
PATH = os.path.dirname(os.path.realpath(__file__))
OUT_PATH = os.path.join(PATH,'output/')
if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)

class myThread (threading.Thread):
    def __init__(self, threadID, name, file_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.file_list = file_list
    
    def getImageName(self, filename):
        index = len(filename)-1
        while not filename[index]=='\\' and index>0:
            index=index-1
        index_dot = filename[-5:].find('.')
        return filename[index:index_dot-5]
    
    def process(self, image, filename):
        width, length = image.size
        min_length = min(width, length)
        # # TEST CROP ONLY
        # x=image.crop((333,253,1285,1553))
        # x.save(filename+'.jpg','JPEG')
        # return
        # #TEST END
        current_length = 32
        while min_length//current_length>32:
            current_length=current_length*2
        
        count =0
        while current_length<=min_length:
            left_bound = width-current_length
            up_bound = length-current_length
            multi_num = (length//current_length+1)*(width//current_length+1)
            if multi_num>10:
                multi_num=10
            pos = [(np.random.randint(left_bound) if left_bound!=0 else 0,np.random.randint(up_bound) if up_bound!=0 else 0) for i in range(multi_num)]
            pos = [(i[0],i[1],i[0]+current_length,i[1]+current_length) for i in pos]
            for i in pos:
                region = image.crop(i)
                region.thumbnail((32,32,))
                region.save(filename+'_X'+str(current_length)+'_'+str(count)+'.jpg','JPEG')
                count=count+1
            current_length = current_length*4

    def run(self):
        for i in self.file_list:
            try:
                imageIn=Image.open(i)
            except IOError:
                print("cannot convert", imageIn)
            self.process(imageIn,OUT_PATH+self.getImageName(i))
                

# Get picture filenames
pic_names = []
pic_end = {'jpg','png','JPG','PNG','jpeg','JPEG'}
for i in os.walk(PATH):
    for j in i[2]:
        if (j[-3:] in pic_end) or (j[-4:] in pic_end):
            pic_names.append(os.path.join(i[0],j))
random.shuffle(pic_names)

threads = []
batch_size = len(pic_names)//THREAD_NUM
# if batch_size<1:
#     THREAD_NUM=0
#     batch_size = len(pic_names)
for i in range(THREAD_NUM):
    start_index = batch_size*i
    end_index = (start_index+batch_size) if not i==THREAD_NUM-1 else len(pic_names)
    threads.append(myThread(i,'thread'+str(i),pic_names[start_index:end_index]))
for i in threads:
    i.start()
for i in threads:
    i.join()