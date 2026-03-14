# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Implement an image classification system using a pre-trained VGG19 convolutional neural network with transfer learning in PyTorch to classify images from a given dataset. The model should load and preprocess images, train on the training dataset, evaluate performance using accuracy, confusion matrix, and classification report, and predict the class label for unseen test images.


## DESIGN STEPS
### STEP 1:
Load the image dataset using PyTorch ImageFolder, apply transformations such as resizing and tensor conversion, and create DataLoaders for training and testing.

### STEP 2:
Load the pre-trained VGG19 model and modify the final fully connected layer to match the number of classes in the dataset for transfer learning.

### STEP 3:

Train the model using an optimizer and loss function, evaluate performance on the test dataset, generate a confusion matrix and classification report, and perform predictions on sample images.

## PROGRAM

```python
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
train_model(model, train_loader, test_loader, num_epochs=10)
```


```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot


### Confusion Matrix





### Classification Report


### New Sample Prediction


## RESULT
The image classification model using transfer learning with VGG19 in PyTorch was successfully trained and tested. The model classified the images from the dataset and its performance was evaluated using accuracy, confusion matrix, and classification report, showing effective prediction of the image classes.
