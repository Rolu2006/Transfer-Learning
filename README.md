# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Implement an image classification system using a pre-trained VGG19 convolutional neural network with transfer learning in PyTorch to classify images from a given dataset. The model should load and preprocess images, train on the training dataset, evaluate performance using accuracy, confusion matrix, and classification report, and predict the class label for unseen test images.
<img width="1360" height="760" alt="Screenshot 2026-03-14 215307" src="https://github.com/user-attachments/assets/04bfad0e-80ba-4800-b7d8-8021a959d470" />



## DESIGN STEPS
### STEP 1:
Load the image dataset using PyTorch ImageFolder, apply transformations such as resizing and tensor conversion, and create DataLoaders for training and testing.

### STEP 2:
Load the pre-trained VGG19 model and modify the final fully connected layer to match the number of classes in the dataset for transfer learning.

### STEP 3:

Train the model using an optimizer and loss function, evaluate performance on the test dataset, generate a confusion matrix and classification report, and perform predictions on sample images.

## PROGRAM


# Load Pretrained Model and Modify for Transfer Learning

```
model = models.vgg19(pretrained=True)
```

# Modify the final fully connected layer to match the dataset classes

```
num_classes = len(train_dataset.classes)

model.classifier[6] = nn.Linear(4096, num_classes)
```

# Include the Loss function and optimizer
```
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)
```



# Train the model
```

## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:  Somalaraju Rohini      ")
    print("Register Number:    212224240156   ")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
train_model(model, train_loader, test_loader, num_epochs=10)
```



## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

<img width="1151" height="854" alt="Screenshot 2026-03-17 161052" src="https://github.com/user-attachments/assets/ff753b03-8d80-4d67-8929-40e9e8571c2c" />



### Confusion Matrix

<img width="1044" height="723" alt="Screenshot 2026-03-17 161116" src="https://github.com/user-attachments/assets/32bfc4ab-343f-480b-b22f-b8d738b72e06" />




### Classification Report




<img width="699" height="234" alt="Screenshot 2026-03-17 161218" src="https://github.com/user-attachments/assets/eed4610a-d462-43e7-ac35-cc8d407aba25" />


### New Sample Prediction



<img width="487" height="610" alt="Screenshot 2026-03-17 161252" src="https://github.com/user-attachments/assets/b63e7692-8b40-479e-a15f-35d8501d4d2d" />


## RESULT
The image classification model using transfer learning with VGG19 in PyTorch was successfully trained and tested. The model classified the images from the dataset and its performance was evaluated using accuracy, confusion matrix, and classification report, showing effective prediction of the image classes.
