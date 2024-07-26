
# Setting Up Raspberry Pi for Model Deployment

This guide will help you set up a Raspberry Pi with the 64-bit Raspbian OS and prepare it for deploying and running machine learning models.

## Steps

### 1. Install 64-bit Raspbian OS

1. Download and install the Raspberry Pi Imager.
2. Use the Raspberry Pi Imager to install the full version of the 64-bit Raspbian OS on your Raspberry Pi.

### 2. Install Required Packages

Before creating the virtual environment, install the following packages:

```sh
sudo apt update
sudo apt install -y numpy pillow torchvision
sudo apt install -y python3-picamera2
sudo apt install -y python3-opencv
sudo apt install -y opencv-data
```

### 3. Connect Camera and Enable Interfaces

1. Connect the camera to your Raspberry Pi.
2. Enable the camera and other necessary interfaces:
   - Open the Raspberry Pi Configuration tool (found in the Preferences menu).
   - Enable the camera and other required interfaces (I2C, SPI, etc.).
3. Reboot the Raspberry Pi.

Check if the camera is working using the following command:

```sh
libcamera-still -o test.jpg
```

### 4. Create and Activate Virtual Environment

1. Create a new folder and open a terminal in that folder.
2. Create a virtual environment with the system site packages inherited:

```sh
python3 -m venv --system-site-packages myenv
source myenv/bin/activate
```

### 5. Install Additional Python Packages

Inside the virtual environment, install the following packages:

```sh
pip3 install torch-pruning optimum-quanto
```

### 6. Train your model on GPU and save the .pth file to the folder containing venv in the raspberry pi.

### 7. Modify and Run the Script

1. Make changes to your script if required (hyperparameters for pruning will vary from model to model).
2. Set the correct model path in your code.
3. Ensure you are using the same test transformations and input size in `script.py` on the Raspberry Pi.

Run your script with:

```sh
python3 script.py
```
