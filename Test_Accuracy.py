import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import onnxruntime as ort
import numpy as np

batch_size = 64

#print(torch.__version__)
#print(torch.cuda.is_available())

transform = transforms.Compose([
    transforms.Resize((80,80)),
    transforms.ToTensor(),
    transforms.Normalize([.5], [.5])
])

#loads the datasets and models for testing
test_Both = torch.load('Datasets/BothTest.pth')
Both_loader = DataLoader(dataset=test_Both, batch_size=batch_size, shuffle=True)

#Pytorch
PytorchModel = torchvision.models.resnet18().to('cuda')
PytorchModel.load_state_dict(torch.load("Models/PytorchModel.pth"))
PytorchModel.eval()

#Onnx
OnnxModel = ort.InferenceSession("Models/OnnxModel.onnx", providers=['CUDAExecutionProvider'])


print('GPU testing\n')
#Test Pytorch Model

correct = 0
total = 0
preloaded_images = []
preloaded_labels = []

#preload images to avoid memory overhead
for images, labels in Both_loader:
    preloaded_images.append(images.to('cuda'))
    preloaded_labels.append(labels.to('cuda'))

start_time = time.time()
with torch.no_grad():
    for images, labels in zip(preloaded_images, preloaded_labels):
        outputs = PytorchModel(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
end_time = time.time()
accuracy = 100 * correct / total
print(f"Accuracy on the Both set by the PytorchModel: {accuracy:.2f}%")
print("Time completed: " + str(end_time - start_time))

#Test Onnx Model

correct = 0
total = 0
preloaded_images = []
preloaded_labels = []

for images, labels in Both_loader:
    preloaded_images.append(images.cpu().numpy())
    preloaded_labels.append(labels.cpu())

start_time = time.time()
for images, labels in zip(preloaded_images, preloaded_labels):
    input_name = OnnxModel.get_inputs()[0].name
    outputs = OnnxModel.run(None, {input_name: images})
    predicted = np.argmax(outputs[0], axis=1)
    total += labels.size(0)  # Total samples
    correct += np.sum(predicted == labels.numpy())
end_time = time.time()

accuracy = 100 * correct / total  # Calculate accuracy
print(f"Accuracy on the Both set by the OnnxModel: {accuracy:.2f}%")
print("Time completed: " + str(end_time - start_time))


print('\nCPU Testing\n')

#Pytorch
PytorchModel = torchvision.models.resnet18().to('cpu')
PytorchModel.load_state_dict(torch.load("Models/PytorchModel.pth"))
PytorchModel.eval()

#Onnx
OnnxModel = ort.InferenceSession("Models/OnnxModel.onnx", providers=['CPUExecutionProvider'])

#Test Pytorch Model

correct = 0
total = 0
preloaded_images = []
preloaded_labels = []

#preload images to avoid memory overhead
for images, labels in Both_loader:
    preloaded_images.append(images.to('cpu'))
    preloaded_labels.append(labels.to('cpu'))

start_time = time.time()
with torch.no_grad():
    for images, labels in zip(preloaded_images, preloaded_labels):
        outputs = PytorchModel(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
end_time = time.time()
accuracy = 100 * correct / total
print(f"Accuracy on the Both set by the PytorchModel: {accuracy:.2f}%")
print("Time completed: " + str(end_time - start_time))

#Test Onnx Model

correct = 0
total = 0
preloaded_images = []
preloaded_labels = []

for images, labels in Both_loader:
    preloaded_images.append(images.cpu().numpy())
    preloaded_labels.append(labels.cpu())

start_time = time.time()
for images, labels in zip(preloaded_images, preloaded_labels):
    input_name = OnnxModel.get_inputs()[0].name
    outputs = OnnxModel.run(None, {input_name: images})
    predicted = np.argmax(outputs[0], axis=1)
    total += labels.size(0)  # Total samples
    correct += np.sum(predicted == labels.numpy())
end_time = time.time()

accuracy = 100 * correct / total  # Calculate accuracy
print(f"Accuracy on the Both set by the OnnxModel: {accuracy:.2f}%")
print("Time completed: " + str(end_time - start_time))