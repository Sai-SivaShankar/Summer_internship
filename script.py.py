import torch
import torch.nn as nn
import torchvision.transforms as transforms
from picamera2 import Picamera2
import cv2
import numpy as np

# Load the quantized MobileNetV2 model
model_quantized = torch.load('/home/pi/New/int8_mobilenet_64.pth', map_location=torch.device('cpu'))
model_quantized.eval()

# Define the MNIST normalization
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale images to 3 channels
    transforms.Resize((64, 64)),  # Resize to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Function to preprocess the image
def preprocess_image(image):
    return transform(image).unsqueeze(0)

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (64, 64)}))
picam2.start()

cv2.startWindowThread()

while True:
    # Capture image as a numpy array
    frame = picam2.capture_array()

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocess the image
    preprocessed_image = preprocess_image(gray_frame)

    # Run inference
    with torch.no_grad():
        predictions = model_quantized(preprocessed_image)

    # Get the class with the highest probability
    _, predicted_class = predictions.max(1)
    predicted_class = predicted_class.item()

    # Display the result on the frame
    cv2.putText(frame, f'Class: {predicted_class}', (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # Show the frame
    cv2.imshow('Camera', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
