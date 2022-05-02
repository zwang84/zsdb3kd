import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


class ModelWrapper(object):
    def __init__(self,model, bounds, num_classes):
        self.model = model
        self.model.eval()
        self.bounds = bounds
        self.num_classes = num_classes
        self.num_queries = 0
    
    def predict(self,image):
        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)
        output = self.model(image)
        self.num_queries += 1
        return output
 
    def predict_prob(self,image):
        with torch.no_grad():
            image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
            if len(image.size())!=4:
                image = image.unsqueeze(0)
            output = self.model(image)
            self.num_queries += image.size(0)
        return output
    
    def predict_label(self, image, batch=False):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)

        with torch.no_grad():
            output = self.model(image)
            self.num_queries += image.size(0)

        _, predict = torch.max(output.data, 1)
        if batch:
            return predict
        else:
            return predict[0]
    
    def predict_ensemble(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
            output.zero_()
            for i in range(10):
                output += self.model(image)
                self.num_queries += image.size(0)

        _, predict = torch.max(output.data, 1)
        
        return predict[0]

    def get_num_queries(self):
        return self.num_queries

    def get_gradient(self,loss):
        loss.backward()


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(4*4*50, 200)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(200, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        feature = output.view(-1, 4*4*50)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        
        return output 


class LeNet5Half(nn.Module):

    def __init__(self):
        super(LeNet5Half, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(10, 25, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(4*4*25, 100)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, img, out_feature=False):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        feature = output.view(-1, 4*4*25)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        
        return output 


class LeNet5Fifth(nn.Module):

    def __init__(self):
        super(LeNet5Fifth, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(4, 10, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(4*4*10, 40)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(40, 10)

    def forward(self, img, out_feature=False):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        feature = output.view(-1, 4*4*10)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        
        return output 


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(4096),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(4096),
        )
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x, out_feature=False):
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        x = self.fc1(feature)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class AlexNetHalf(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNetHalf, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 2 * 2, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(2048),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(2048),
        )
        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x, out_feature=False):
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        x = self.fc1(feature)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class AlexNetQuarter(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNetQuarter, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(1024),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(1024),
        )
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x, out_feature=False):
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        x = self.fc1(feature)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
