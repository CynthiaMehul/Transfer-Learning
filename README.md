# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.

## Problem Statement and Dataset
Image classification from scratch requires large datasets and extensive training. Transfer Learning allows us to use pre-trained deep learning models (such as VGG-19 trained on ImageNet) and fine-tune them for our custom dataset, reducing computational cost and training time. In this experiment, we apply transfer learning using VGG-19 for a binary classification dataset, modifying the final layer to match the number of target classes.

## DESIGN STEPS
### STEP 1:
Import the required libraries, define dataset path and apply image transformations.

### STEP 2:
Load the pre-trained VGG-19 model and replace the final fully connected layer to match the number of classes in our dataset.

### STEP 3:
Define the loss function (BCEWithLogitsLoss) and optimizer (Adam) and train your model then check for results.

## PROGRAM
```python
# Load Pretrained Model and Modify for Transfer Learning
dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
model = models.vgg19(pretrained=True)

# Modify the final fully connected layer to match the dataset classes
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 1)

# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)

# Train the model
## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1) # Move data to device and adjust shape
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1) # Move data to device and adjust shape
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:Cynthia Mehul J")
    print("Register Number:212223240020")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
train_model(model, train_loader,test_loader,num_epochs=10)
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

<img width="692" height="716" alt="image" src="https://github.com/user-attachments/assets/d943b7b7-0578-4a47-83b6-49cb4d459f7a" />

### Confusion Matrix

<img width="639" height="568" alt="image" src="https://github.com/user-attachments/assets/aa4c5d74-15da-4879-88f4-17a744995192" />

### Classification Report

<img width="840" height="384" alt="image" src="https://github.com/user-attachments/assets/16756930-81d6-4416-9a5c-402974d90c9f" />

### New Sample Prediction

<img width="444" height="419" alt="image" src="https://github.com/user-attachments/assets/f4300a05-bdb4-4c43-b32b-59ab2a0a99e0" />

## RESULT
Therefore, transfer learning using the VGG-19 architecture was successfully implemented for classification.
