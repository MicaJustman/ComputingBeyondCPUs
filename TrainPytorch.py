import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

learning_rate = 1e-3
batch_size = 32
num_epochs = 18

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((80,80)),
    transforms.ToTensor(),
    transforms.Normalize([.5], [.5])
])

train_set = torch.load('Datasets/BothTrain.pth')
test_set = torch.load('Datasets/BothTest.pth')

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = torchvision.models.resnet18()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_correct = 0.0
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        output_idx = torch.argmax(output, dim=1)
        total_correct += (labels == output_idx).sum().item()
        optimizer.zero_grad()
        loss = criterion(output, labels)
        running_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()

    print("Epoch:" + str(epoch) + "  loss:" + str(running_loss/train_set.__len__()) + "  Acc:" + str(total_correct/train_set.__len__()))

torch.save(model.state_dict(), 'Models/PytorchModel.pth')

