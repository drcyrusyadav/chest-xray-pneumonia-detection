import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

base_dir = r'C:\Users\User\Downloads\archive\chest_xray'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = ImageFolder(base_dir + r'\train', transform=transform)
val_data = ImageFolder(base_dir + r'\val', transform=transform)
test_data = ImageFolder(base_dir + r'\test', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images).squeeze()
        loss = loss_fn(outputs, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1} done, Loss: {total_loss/len(train_loader):.4f}')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images).squeeze()
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f'\nTest Accuracy: {correct/total*100:.2f}%')
torch.save(model.state_dict(), 'pneumonia_model.pth')
print('Model saved!')
