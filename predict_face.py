import torch
import cv2
from torch.autograd import Variable
from neural_network import MyNetwork
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

idx_to_label = {
        0: 'chloe',
        1: 'toby',
        2: 'adam'
    }

# NN = torch.load('/content/drive/My Drive/classifier_10_epochs.pt')
# NN.eval()
# print("> Done Eval")

name = "adam"
path = "../face_detector_data/" + name + "/" + name+ "_face10.png"
print(name)
read_image = cv2.imread(path, cv2.COLOR_BGR2GRAY)
resized_read_image = cv2.resize(read_image, (32, 32), interpolation = cv2.INTER_AREA)
image_tensor = torch.Tensor(resized_read_image)

test_input = Variable(image_tensor)
test_input = test_input.float()
test_input = test_input.unsqueeze(0)

# inputting this gave same output everytime
# dataiter = iter(training_generator)
# images, labels = dataiter.next()

NN = MyNetwork()
NN.load_state_dict(torch.load("/Users/tobycourtis/Downloads/10_epochs_state_dict.pt"))
NN.eval()

output = NN(test_input)
print(len(output))
print(output)

print("-----")
print("-----")
print("-----")
# input_for_NN = image_tensor.to(device)
# output = NN(input_for_NN)

prediction = np.argmax(output.data[0].numpy())
print("Prediction: ", prediction)
# print("Max(1)[1]: ", output.max(1)[1])
# _, predicted = torch.max(output, 1)
# print("> Predicted: {}".format(predicted[0]))
