# Introduction
Scalable classifier

This project makes use of the NCAPI v2 FIFOs and is a good example of both how to scale and thread an NCSDK application.

The provided Makefile does the following:
1. Builds caffe googlenet graph file from the caffe/GoogLeNet directory in the repository.
2. Copies the built NCS graph file from the GoogLeNet directory to the project base directory
3. Runs the provided python program which creates a GUI window that shows the objects and their classifications

To run from the commandline type the following command
```code
python3 classifier_scalable.py
```
To get a list of commandline options type the following command
```code
python3 classifier_scalable.py help
```



# Prerequisites
This program requires:
- 1 or more NCS devices
- NCSDK 2.04 or greater
- OpenCV 3.3 with Video for Linux (V4L) support and associated Python bindings*.

*It may run with older versions but you may see some glitches such as the GUI Window not closing when you click the X in the title bar, and other key binding issues.


Note: The OpenCV version installed with some earlier versions of the ncsdk do <strong>not</strong> provide V4L support.  To install a compatible version of OpenCV for this application and remove older versions installed by the ncsdk you can run the following command from the app's base directory:
```
   make install-reqs
```   
Note: All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.


This applications is similar to the apps/video_objects application in the ncappzoo, but it adds threads for reading from the video file and processing SSD MobieNet inferences.  This change increases performance on most systems pushing closer to the  max frame rate of the neural network.

# Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

## make help
Shows available targets.

## make all
Builds and/or gathers all the required files needed to run the application except building and installing opencv (this must be done as a separate step with 'make install-reqs'.)

## make install-reqs
Installs required packages that aren't installed as part of the NCSDK installation.  For this app that is OpenCV and python bindings.  This will remove the version of OpenCV that was installed with the NCSDK and builds and installs a compatible version of OpenCV 3.3 for this app. This will take a while to finish. Once you have done this on your system you shouldn't need to do it again.

## make run
Runs the provided python program which shows the video stream along with the object boxes and classifications.

## make clean
Removes all the temporary files that are created by the Makefile
