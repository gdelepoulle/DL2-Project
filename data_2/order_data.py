import numpy as np
np.random.seed(123)
import os
import shutil

seed= 1234

WORD2VECPATH    = "../data/class_vectors.npy"
DATAPATH        = "data/DatasetA_train_20180813/train/"
MODELPATH       = "../model/"

np.random.seed(1234)

global train_classes
with open('data/DatasetA_train_20180813/label_list.txt', 'r') as infile:
    name_classes = [str.strip(line).split('\t')[0] for line in infile]
    #Lets take 30 as Zsl classes
global zsl_classes
indexes = np.arange(0, len(name_classes))
zsl_indexes = np.random.choice(indexes, size=30, replace=False)
zsl_classes = np.array(name_classes)[zsl_indexes]
train_classes= []
for i,obj in enumerate(name_classes):
    if not obj in zsl_classes:
        train_classes.append(obj)

train_classes =  np.array(train_classes)



############# CREATE FOLDER ################
path = 'data/ordered_data/'
try:
    os.mkdir(path)
    os.mkdir(path+'training/')
    os.mkdir(path+'zeroshot/')
except OSError:
    print ("Folder %s already exists" % path)
else:
    print ("Successfully created the directory %s " % path)



try:
    for label in train_classes:
        os.mkdir(path+'training/'+label)
    for label in zsl_classes:
        os.mkdir(path+'zeroshot/'+label)
except OSError:
    print("Folders alreday exists")

############ CREATE DICTIONNARY ############
dic_train = {}
dic_zeroshot = {}
for label in train_classes:
    dic_train[label]= []

for label in zsl_classes:
    dic_zeroshot[label]= []


with open('data/DatasetA_train_20180813/train.txt', 'r') as infile:
    name_images = [str.strip(line).split('\t') for line in infile]

for item in name_images:
    if item[1] in train_classes:
        dic_train[item[1]].append(item[0])
    else:
        dic_zeroshot[item[1]].append(item[0])

try:
    for label, images in dic_train.items():
        for image in images:
            shutil.copyfile(DATAPATH+image, path+'training/'+label+'/'+image)
    for label, images in dic_zeroshot.items():
        for image in images:
            shutil.copyfile(DATAPATH+image, path+'zeroshot/'+label+'/'+image)

except OSError:
    print("Problem in copy")
else:
    print("Success")
    




