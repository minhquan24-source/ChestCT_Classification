import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

class CancerDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label

def train(train_loader, model, epochs, learning_rate):
    '''
    Train the model on the training set.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}', end='\r')
                running_loss = 0.0

        epoch_duration = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{epochs}] completed in {epoch_duration:.2f} seconds')

    return model

if __name__ == '__main__':
    train_dir = './data/curated/train'
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    train_files = [(os.path.join(train_dir, c, img), i) for i, c in enumerate(classes) for img in os.listdir(os.path.join(train_dir, c))]

    validation_dir = './data/curated/valid'
    val_files = [(os.path.join(validation_dir, c, img), i) for i, c in enumerate(classes) for img in os.listdir(os.path.join(validation_dir, c))]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels for ViT
        transforms.RandomRotation(15),  # Rotate images randomly within 15 degrees
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color properties randomly
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),  # Crop to 224x224 with random scale
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_data = CancerDataset(train_files, transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

    # Load the pretrained ViT model
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    # Modify the classifier layer to match the number of classes
    num_classes = len(classes)  # Assuming 'classes' contains the list of class labels
    # Access and replace the classification layer
    if hasattr(model, 'fc'):  # If 'fc' layer exists
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):  # If 'classifier' layer exists
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
    elif hasattr(model, 'heads'):  # If 'heads' layer exists
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
    else:
        raise AttributeError("Could not find the classification layer in the ViT model.")


    model = train(train_loader, model, epochs=10, learning_rate=0.001)

    val_data = CancerDataset(val_files, transform=transform)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=1)
            all_preds.extend(predicted_class.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
