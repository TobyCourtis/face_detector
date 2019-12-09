import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.Linear(in_features=273*273*3, out_features=512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=512, out_features=32))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=32, out_features=512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=512, out_features=273*273*3))
        layers.append(nn.Sigmoid())
        self.layers = layers

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x

    # encode (flatten as linear, then run first half of network)
    def encode(self, x):
        x = x.view(x.size(0), -1)
        for i in range(4):
            x = self.layers[i](x)
        return x

    # decode (run second half of network then unflatten)
    def decode(self, x):
        for i in range(4,8):
            x = self.layers[i](x)
        x = x.view(x.size(0), 273, 273, 3)
        return x
