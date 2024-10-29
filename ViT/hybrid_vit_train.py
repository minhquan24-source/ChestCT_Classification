import os
from hybrid_vit import HybridViT, train_step
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
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
        
        # Load the image
        image = Image.open(image_path)  # Convert to grayscale if needed

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label

def train(train_loader, epochs, learning_rate, depth):
    '''
    Description: Train the model on the training set

    Parameters:
    - train_loader (DataLoader): The DataLoader object for the training set
    - epochs (int): The number of epochs to train the model
    - learning_rate (float): The learning rate for the optimizer
    - depth (int): The depth of the model

    Returns:
    - model (HybridViT): The trained model
    '''

    model = HybridViT(
        image_size=128,
        patch_size=16,
        dim=512,
        depth=depth,
        heads=6,
        mlp_dim=2048,
        num_class=4,
        channels=1,
        GPU_ID=None # Specify GPU ID if available
    )

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Typically use cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # Start counting time for each epoch
        epoch_start_time = time.time()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            loss_item = train_step(inputs, labels, model, criterion, optimizer)

            running_loss += loss_item

            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}', end='\r')
                running_loss = 0.0

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch [{epoch+1}/{epochs}] completed in {epoch_duration:.2f} seconds')
    
    return model

if __name__ == '__main__':
    train_dir = './data/curated/train'

    classes = os.listdir(train_dir)
    if '.DS_Store' in classes:
        classes.remove('.DS_Store')
    train_files = []

    for i, c in enumerate(classes):
        images = os.listdir(os.path.join(train_dir, c))
        train_files.extend([(os.path.join(train_dir, c, img), i) for img in images])

    validation_dir = './data/curated/valid'
    val_files = []

    for i, c in enumerate(classes):
        images = os.listdir(os.path.join(validation_dir, c))
        val_files.extend([(os.path.join(validation_dir, c, img), i) for img in images])

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize([0.5], [0.5])  # Normalize the grayscale images
    ])

    # Define the DataLoader
    train_data = CancerDataset(train_files, transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

    # Train the model
    model = train(train_loader, epochs=10, learning_rate=0.001, depth=4)

    # Validate the model
    val_data = CancerDataset(val_files, transform=transform)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=4)

    # Initialize lists to store all true labels and predicted labels
    all_preds = []
    all_labels = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)  # Forward pass
            probabilities = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
            # Get the predicted class
            predicted_class = torch.argmax(probabilities, dim=1)
            
            # Store predictions and true labels
            all_preds.extend(predicted_class.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Now all_preds and all_labels have the predictions and true labels, respectively.
    # Calculate the accuracy
    correct_predictions = sum(p == t for p, t in zip(all_preds, all_labels))
    total_predictions = len(all_labels)
    accuracy = correct_predictions / total_predictions

    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the graph
    plt.savefig('confusion_matrix.png')