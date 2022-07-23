import math

import torch.nn as nn
import torch.nn.init as init

# Define VGG model architecture
class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
        )
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.linear(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

# Define VGG model with IC layer
class IC_layer(nn.Module):
    def __init__(self, planes, p=0.05, dim ='2d'):
        super(IC_layer, self).__init__()
        if dim == '2d':
            self.bn = nn.BatchNorm2d(planes)
        elif dim == '1d':
            self.bn = nn.BatchNorm1d(planes)
        
        self.dropout = nn.Dropout(p=p)
    
    def forward(self, x):
        return self.dropout(self.bn(x))
    
class VGG_IC(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes):
        super(VGG_IC, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            IC_layer(512, dim='1d'),
            nn.Linear(512, 512),
            nn.ReLU(True),
            IC_layer(512, dim='1d'),
            nn.Linear(512, 512),
            nn.ReLU(True),
            IC_layer(512, dim='1d'),
        )
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.linear(x)
        return x


def make_layers_ic(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [IC_layer(in_channels), conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# model functions
def vgg16_bn(num_classes):
    print("INFO: Creating VGG-16 Model")
    model = VGG(make_layers(cfg['D'], batch_norm=True), num_classes)
    model.model_name = "vgg16"
    return model 

def vgg16_bn_ic(num_classes):
    print("INFO: Creating VGG-16 Model with IC layer")
    model = VGG_IC(make_layers_ic(cfg['D']), num_classes)
    model.model_name = "vgg16_ic"
    return model 

