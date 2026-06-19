import torch
from torchvision import models, transforms
from PIL import Image
import os

def load_model(path_to_weights, num_classes=3):
    # Load the pretrained ResNet18 model
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    # Replace the fully connected layer with onefthat matches the number of classes
    model.fc = torch.nn.Linear(num_features, num_classes)
    
    # Load the saved weights
    model.load_state_dict(torch.load(path_to_weights, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def predict_image(image_path, model, device):
    ''' Predict the class of an image using a trained model. '''
    # Define the transformation including normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust these values if your training normalization was different
    ])

    # Load the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move image to the correct device
    image = image.to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        index = predicted.item()

    return index

def main():
    # Path to the trained model weights
    model_path = 'model.pth'
    
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model with the correct number of classes (4 instead of 3)
    model = load_model(model_path, num_classes=4)  # Change num_classes to 4
    model = model.to(device)

    # Path to the image you want to predict
    image_path = 'window_131.jpg'

    # Predict the image
    class_index = predict_image(image_path, model, device)
    classes = ['Earth', 'Mars', 'Mercury', 'Moon']  # Correct the list of classes to include all four
    
    # Print the prediction
    print(f'The image is predicted to be: {classes[class_index]}')

if __name__ == "__main__":
    main()