# Automatic flare video annotation module with Graphical User Interface (GUI)

A GUI tool that performs auto annotation for flare stack images. The tool does the following:

* Shows the user a log with the auto-annotation progress.
* Performs the auto-annotation for detection.
* Saves both the raw labels in YOLO detection format and the illustrated detection images in two subfolders in the output saving directory.
* Performs the auto-annotation for segmentation.
* Saves both the raw labels in YOLO segmentation format and the illustrated segmentation images in two subfolders in the output saving directory.
* Features a "Results Visualizer" window that shows illustrations of original images along with their detection and segmentation images.
* Features a "Measurements" tab in the "Results Visualizer" to illustrate some quantitative results related to flame and smoke appearance, size, and orientation.




## Main window

!(MainWindowSample.jpg)



## Results Visualizer

!(ResVisualizerSample.jpg)




## Saved Output

!(GeneratedOutputSample.jpg)




## Installation

Install Anaconda and Python and install the libraries detailed in "requirements.txt" file. Running "GUImain.py" runs the GUI tool.


