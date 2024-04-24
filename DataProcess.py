import fnmatch
import os
import shutil
import torch
import torchvision
from torch.utils.data import random_split
from torchvision.transforms import transforms


'''
Data
    Emotion-both
        angry
        happy
        neutral
        sad
    Emotion-open
        angry
        happy
        neutral
        sad
'''

#faces is the name of my raw dataset
count = 0
for path, dirs, fileList in os.walk('faces'):
    print(path)
    for file in fileList:
        count += 1
        if fnmatch.fnmatch(file, '*open*'):
            if fnmatch.fnmatch(file, '*happy*'):
                shutil.copy(path + '/' + file, 'Data/Emotion-open/happy')
                shutil.copy(path + '/' + file, 'Data/Emotion-both/happy')
            elif fnmatch.fnmatch(file, '*sad*'):
                shutil.copy(path + '/' + file, 'Data/Emotion-open/sad')
                shutil.copy(path + '/' + file, 'Data/Emotion-both/sad')
            elif fnmatch.fnmatch(file, '*angry*'):
                shutil.copy(path + '/' + file, 'Data/Emotion-open/angry')
                shutil.copy(path + '/' + file, 'Data/Emotion-both/angry')
            elif fnmatch.fnmatch(file, '*neutral*'):
                shutil.copy(path + '/' + file, 'Data/Emotion-open/neutral')
                shutil.copy(path + '/' + file, 'Data/Emotion-both/neutral')
        else:
            if fnmatch.fnmatch(file, '*happy*'):
                shutil.copy(path + '/' + file, 'Data/Emotion-both/happy')
            elif fnmatch.fnmatch(file, '*sad*'):
                shutil.copy(path + '/' + file, 'Data/Emotion-both/sad')
            elif fnmatch.fnmatch(file, '*angry*'):
                shutil.copy(path + '/' + file, 'Data/Emotion-both/angry')
            elif fnmatch.fnmatch(file, '*neutral*'):
                shutil.copy(path + '/' + file, 'Data/Emotion-both/neutral')

print(count)
#transform to standardize photo size

transform = transforms.Compose([
    transforms.Resize((80,80)),
    transforms.ToTensor(),
    transforms.Normalize([.5], [.5])
])

#creates the two datasets using torchvision and saves them to Datasets directory
dataset = torchvision.datasets.ImageFolder(root='Data/Emotion-both', transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

torch.save(train_set, 'Datasets/BothTrain.pth')
torch.save(test_set, 'Datasets/BothTest.pth')

dataset = torchvision.datasets.ImageFolder(root='Data/Emotion-open', transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

torch.save(train_set, 'Datasets/OpenTrain.pth')
torch.save(test_set, 'Datasets/OpenTest.pth')