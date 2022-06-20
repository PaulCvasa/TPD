- User Manual

-- General Information

Traffic Participants Detection (TPD) is a Python application designed to assist
drivers in predicting a front collision situation by detecting each traffic participant,
regardless of it being a car, a motorcycle, a truck or a pedestrian. The purpose of
it is also to equip older cars who do not have an ADAS (advanced driver assistance
system) in them with this important function. The system uses a Tensorflow model
to detect the participants and afterwards it computes if it’s in the ego vehicle path
or not along with it’s approximate distance to collision.

-- System Overview

The system requires a laptop or a small computer with a capable CPU (a graphics
card is recommended but not mandatory), a webcam (a HD webcam is recom-
mended for better accuracy of detection) and a vehicle in which it can be placed.
If the computer has a graphics card equipped, it will also need to have installed
the NVIDIA GPU Computing Toolkit CUDA with version 11.2 or newer and the
NVIDIA CUDNN driver version 8.1 or newer, so the Tensorflow will see and use
the GPU instead of the CPU.

-- Getting Started

In order to install the TPD application, you will need to follow this steps in order:
• Install Python 3.9.13 from this link, scroll at the end of the page and select
the Windows installer (64-bit) or Windows installer (32-bit) depending on
your type of system:
https://www.python.org/downloads/release/python-3913/

• While installing Python, make sure to check the ”Add Python to environment
variables” option

• After installation has finished successfully, make sure you have the latest
version of pip by using this command:
python -m pip install –upgrade pip

• Clone the TPD GitHub repository:
https://github.com/PaulCvasa/TPD.git

• Install all the required libraries for the application by opening a command
prompt in the TPD folder and running this command:
py -m pip install -r requirements.txt

• Download the Tensorflow API from the Google Drive shared folder:
https://drive.google.com/file/d/1DUmediZ2HRg4sQrkxwJcq000byVsJqGV/view?
usp=sharing

• Extract the tensorflow folder in the same directory as the TPD.py script

• If the device running the program doesn’t have a GPU, skip the next 10 steps,
and comment this 2 lines from the TPD.py script: line 6 and line 7.

• If the device is equipped with a GPU, download the CUDA Toolkit 11.2 from
here: https://developer.nvidia.com/cuda-11.2.0-download-archive

• In the NVIDIA website link, select the Windows operating system, then the
x86 64 architecture, the version should be 10 and the installer type exe(local).
After that a download button will appear.

• Install the CUDA Toolkit in the following path: C:/Program Files/NVIDIA
GPU Computing Toolkit

• After the setup finished successfully, add this to your system’s envi-
ronment variables PATH: C:/Program Files/NVIDIA GPU Computing
Toolkit/CUDA/v11.2/bin

• Next, Tensorflow also needs a NVIDIA cuDNN library of version
v8.1.0 in order to be more optimized for deep neural networks and
faster in detection. This library can be downloaded from here:
https://developer.nvidia.com/rdp/cudnn-archive

• On that website, scroll down until you find the correct version (”Download
cuDNN v8.1.0 (January 26th, 2021), for CUDA 11.0,11.1 and 11.2”), after
selecting it, in the ”Download cuDNN v8.1.0 (January 26th, 2021), for CUDA
11.0,11.1 and 11.2” section, find the one for Windows (”cuDNN Library for
Windows (x86)”)

• After clicking on that library, a login/register will be required for the ”NVIDIA
Developer Program Membership”, then, a download will start.

• Next step would be to go in C:/Program Files and create a new folder called
”NVIDIA”. Inside that folder create a new one called ”CUDNN” and inside
it a new one called ”v8.1”.

• Unzip the ”cuda” folder contents from the downloaded archive in: C:/Program
Files/NVIDIA/CUDNN/v8.1

• Add the following folder to the system’s environment variables PATH:
C:/Program Files/NVIDIA/CUDNN/v8.1/bin

• Finally, the script TPD.py needs to be executed and the program will run
with no errors.
