import torch
import torch.nn as nn
import torch.utils.data as utils
import cv2
import os
import random
from local_classes import Dataset

########
#print("This is for training the classifier based off my university project")
#https://colab.research.google.com/gist/cwkx/c0e7421f470255bb6536e523dba703b5/coursework-pegasus.ipynb#scrollTo=RGbLY6X-NH4O
########




# Section 1 - Iterate directory and create labels and data lists
training_data_path = "../face_detector_data"
dataset = []
labels = []

for dirname, dirnames, filenames in os.walk(training_data_path):
    for filename in filenames:
        if((filename.find(".png") > -1) | (filename.find(".jpg") > -1)):
            name = str(dirname.split('/')[-1])
            read_image = cv2.imread(dirname+'/'+filename, cv2.COLOR_BGR2GRAY)
            resized_read_image = cv2.resize(read_image, (273, 273), interpolation = cv2.INTER_AREA)
            dataset.append(resized_read_image)
            labels.append(name)

#shuffle lists and dataset/labels are now shuffled in the same order dataset'img' == labels'img label'
z = list(zip(dataset, labels))
random.shuffle(z)
dataset, labels = zip(*z)

# initialise training dataset // could make the above functions and save to pkl if needed
training_set = Dataset(dataset, labels)




# Section 2 - Input data into CNN and TRAIN








# Section 3 - Save classifier

# model = classifier.fit(np.array(data_input),data_input_labels,sample_weight=None)
# print("Saving Classifier...")
# filename = 'classifier_network_type.sav'
# pickle.dump(model, open(filename, 'wb'))
# print("Finished saving Classifier...")



path = "../face_detector_data/toby/toby_face0.png"
image = cv2.imread(path, cv2.COLOR_BGR2GRAY)
scale = 25

while(False):
    #cv2.imshow("test", image)
    width = int(image.shape[1] * (scale/100))
    height = int(image.shape[0] * (scale/100))
    dim = (width, height)
    resized = cv2.resize(image, (273, 273), interpolation = cv2.INTER_AREA)
    cv2.imshow("resized_image", resized)
    k = cv2.waitKey(1)
    if (k%256 == 27): #ESC Pressed - exit
        break

cv2.destroyAllWindows()
