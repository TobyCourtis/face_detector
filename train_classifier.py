import torch
import torch.nn as nn
import torch.utils.data as utils
from torchvision import transforms
import cv2
import os
import random
import numpy as np
from local_classes import Dataset
from neural_network import MyNetwork
from torch.utils import data
#from livelossplot import PlotLosses

########
#print("This is for training the classifier based off my university project")
#https://colab.research.google.com/gist/cwkx/c0e7421f470255bb6536e523dba703b5/coursework-pegasus.ipynb#scrollTo=RGbLY6X-NH4O
########


##############################
# Section 1 - Iterate directory and create labels and data lists
##############################
def generate_dataset():
    training_data_path = "../face_detector_data"
    dataset = []
    labels = []

    idx_to_label = {
        0: 'chloe',
        1: 'toby',
        2: 'adam',
        3: 'max',
        4: 'aaron',
        5: 'alexandra',
        6: 'alvaro',
        7: 'alycia',
        8: 'amanda',
        9: 'amaury',
        10: 'amber',
        11: 'anna'
    }

    rev_dict = {v: k for k, v in idx_to_label.items()}

    pred = 0
    for dirname, dirnames, filenames in os.walk(training_data_path):
        for filename in filenames:
            if((filename.find(".png") > -1) | (filename.find(".jpg") > -1)):
                name = str(dirname.split('/')[-1])
                read_image = cv2.imread(dirname+'/'+filename, cv2.COLOR_BGR2GRAY)
                resized_read_image = cv2.resize(read_image, (32, 32), interpolation = cv2.INTER_AREA)
                dataset.append(resized_read_image)
                labels.append(rev_dict[name])

    z = list(zip(dataset, labels))
    random.shuffle(z)
    dataset, labels = zip(*z)
    return list(dataset), list(labels)
    return dataset, labels

generated = generate_dataset()
dataset = generated[0]
labels  =  generated[1]

print("> Datasets genereated")
print("Dataset Length: {}".format(len(dataset)))
print("Label Length: {}".format(len(labels)))

# initialise training dataset // could make the above functions and save to pkl if needed
training_set = Dataset(dataset, labels)

parameters = {'batch_size': 16, 'shuffle': False, 'num_workers': 6}
training_generator = data.DataLoader(training_set, **parameters)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

train_iterator = iter(cycle(training_generator))





##############################
# Section 2 - Input data into NN from neural_network.py and train
##############################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
N = MyNetwork().to(device)

print("> Number of network parameters {}".format(len(torch.nn.utils.parameters_to_vector(N.parameters()))))

# initialise the optimiser
optimiser = torch.optim.Adam(N.parameters(), lr=0.001)

epoch = 0
max_epochs = 5
print("> Training NN")
while(epoch < 1):
    # arrays for metrics
    logs = {}
    train_loss_arr = np.zeros(0)
    valid_loss_arr = np.zeros(0)

    # iterate over some of the train dateset
    for i in range(1000):
        # x,t = next(train_iterator)
        # x,t = x.to(device), t.to(device)
        for x, t in training_generator:
            x, t = x.to(device), t.to(device)
        optimiser.zero_grad()
        p = N(x)
        loss = ((p-x)**2).mean() # simple l2 loss
        loss.backward()
        optimiser.step()

        train_loss_arr = np.append(train_loss_arr, loss.cpu().data)

    # NOTE: live plot library has dumb naming forcing our 'test' to be called 'validation'
    # liveplot.update({
    #     'loss': train_loss_arr.mean()
    # })
    # liveplot.draw()

    epoch = epoch+1
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss_arr))
    # print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss_arr), "Valid Loss: ", np.mean(valid_loss))

print("> Finished training Neural Network")




##############################
# Section 3 - Save classifier
##############################

classifier_name = "10_epochs"

torch.save(N, "../" + classifier_name + ".pt")
torch.save(N.state_dict(), "../" + classifier_name + "_state_dict" +  ".pt")
print("> Classifier Saved as: " + classifier_name + ".pt")
