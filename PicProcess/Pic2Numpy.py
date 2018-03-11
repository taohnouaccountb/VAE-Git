import threading
import os
import random
import numpy as np
from copy import deepcopy
from PIL import Image

THREAD_NUM = 16
PATH = os.path.dirname(os.path.realpath(__file__))
OUT_PATH = os.path.join(PATH, 'output/')
if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)


class myThread(threading.Thread):
    def __init__(self, threadID, name, file_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.file_list = file_list
        self.image_array = []

    def getImages(self):
        return self.image_array

    def getImageName(self, filename):
        index = len(filename) - 1
        while not filename[index] == '\\' and index > 0:
            index = index - 1
        index_dot = filename[-5:].find('.')
        return filename[index:index_dot - 5]

    def PIL2array(self, img):
        return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

    def run(self):
        for i in self.file_list:
            try:
                imageIn = Image.open(i)
            except IOError:
                print("cannot convert", imageIn)
            imgArr = self.PIL2array(imageIn)
            self.image_array.append(imgArr)

# Get picture filenames
pic_names = []
pic_end = {'jpg', 'png', 'JPG', 'PNG', 'jpeg', 'JPEG'}
for i in os.walk(PATH):
    for j in i[2]:
        if (j[-3:] in pic_end) or (j[-4:] in pic_end):
            pic_names.append(os.path.join(i[0], j))
random.shuffle(pic_names)

threads = []
batch_size = len(pic_names) // THREAD_NUM
for i in range(THREAD_NUM):
    start_index = batch_size * i
    end_index = (start_index + batch_size) if not i == THREAD_NUM - 1 else len(pic_names)
    threads.append(myThread(i, 'thread' + str(i), pic_names[start_index:end_index]))
final_store = []
for i in threads:
    i.start()
for i in threads:
    i.join()
    final_store.extend(i.getImages())
final_store = np.array(final_store)
np.save('./output/data.npy',final_store)