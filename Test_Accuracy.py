import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import onnxruntime as ort
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

batch_size = 375

#print(torch.__version__)
#print(torch.cuda.is_available())
#print(trt.__version__)

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

#ONNX with tensorRT
#tensorRTModel = ort.InferenceSession("Models/OnnxModel.onnx", providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"])

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

#Test TensorRT Model
correct = 0
total = 0

trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
with open("Models/TensorRTModel.trt", "rb") as f:
    engine = trt_runtime.deserialize_cuda_engine(f.read())

# Step 2: Create execution context
context = engine.create_execution_context()

# Step 3: Allocate memory for input and output on CUDA device
input_shape = (375, 3, 80, 80)  # Adjust this to match your model's input shape
output_shape = (375, 1000)  # Adjust based on your model's output shape

# Calculate the total number of elements
total_elements = np.prod(input_shape)
element_size = np.float32(0).nbytes  # 4 bytes for float32
memory_size = total_elements * element_size  # Should be an int
memory_size = int(memory_size)  # Ensure the correct data type
d_input = cuda.mem_alloc(memory_size)

total_elements = np.prod(output_shape)
element_size = np.float32(0).nbytes  # 4 bytes for float32
memory_size = total_elements * element_size  # Should be an int
memory_size = int(memory_size)  # Ensure the correct data type
d_output = cuda.mem_alloc(memory_size)

# Allocate memory on the host (CPU)
h_input = np.zeros(input_shape, dtype=np.float32)  # Input tensor
h_output = np.zeros(output_shape, dtype=np.float32)  # Output tensor


# Step 5: Run inference
results = []

start_time = time.time()
for images, labels in zip(preloaded_images, preloaded_labels):
    h_input = images
    cuda.memcpy_htod(d_input, h_input)
    context.execute_v2([int(d_input), int(d_output)])
    # Transfer output data from device to host
    cuda.memcpy_dtoh(h_output, d_output)
    predicted = np.argmax(h_output, axis=1)  # Get the class with the highest score
    total += labels.size(0)  # Total samples
    correct += np.sum(predicted == labels.numpy())
end_time = time.time()


accuracy = (correct / total) * 100  # Accuracy in percentage
print(f"Accuracy on the dataset with TensorRT model: {accuracy:.2f}%")
print("Time completed: " + str(end_time - start_time))
print(total)


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