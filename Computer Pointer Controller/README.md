# Computer Pointer Controller

In this project we have developed an Intel OpenVINO-based Edge-AI application that can control and move the pointer on computer screen based on the movement of gaze of human eye. This application can take an image, video or webcam feed as input. From the video input, it first detects the presence of face in the input frame. Then it detects the landmark of the face and pose of the head. Finally from the detected landmarks and pose, it estimates the gaze of the eyes. Based on this gaze, computer pointer is controlled. The pipeline flow of the models ar shown below:

![Pipeline flow](https://github.com/arin1405/Intel_Edge_AI_OpenVINO_Projects/blob/master/Computer%20Pointer%20Controller/images/pipeline.png "Pipeline flow")

## Project Set Up and Installation
For successfull development and execution of the project, Intel [OpenVINO](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) needs to be installed. The installation document can be found [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html). I have used OpenVINO 2020.1.033 version for this project.

We need four pretrained models of Intel OpenVINO. These models can be downloaded by model downloader script `downloader.py`.

**1. [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)**
```
python C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.1.033\\deployment_tools\\tools\\model_downloader\\downloader.py --name "face-detection-adas-binary-0001"
```

**2. [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)**
```
python C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.1.033\\deployment_tools\\tools\\model_downloader\\downloader.py --name "head-pose-estimation-adas-0001"
```

**3. [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)**
```
python C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.1.033\\deployment_tools\\tools\\model_downloader\\downloader.py --name "landmarks-regression-retail-0009"
```

**4. [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)**
```
python C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.1.033\\deployment_tools\\tools\\model_downloader\\downloader.py --name "gaze-estimation-adas-0002"
```

Below mentioned Python libraries are required for this project. These are given in `requirements.txt` file:
```
image==1.5.27
ipdb==0.12.3
ipython==7.10.2
numpy==1.17.4
Pillow==6.2.1
requests==2.22.0
virtualenv==16.7.9
```
These need to be installed first:
```
pip3 install -r requirements.txt
```
The folder structure of the project is shown below:

![folder tree](https://github.com/arin1405/Intel_Edge_AI_OpenVINO_Projects/blob/master/Computer%20Pointer%20Controller/images/tree.JPG "folder tree")

The src directory stores all the scripts for this project. The structure of the src folder is shown below:

![src directory](https://github.com/arin1405/Intel_Edge_AI_OpenVINO_Projects/blob/master/Computer%20Pointer%20Controller/images/src_tree.JPG "src tree")

## Demo

To run the program, one needs to follow below mentioned steps:

**Step 1:**
Open a new command prompt window and change the directory to `src`.
cd <project-repo-path>/src

**Step 2:**
Execute the main.py:
```
python main.py -fd "..\\intel\\face-detection-adas-binary-0001\\FP32-INT1\\face-detection-adas-binary-0001" -fld "..\\intel\\landmarks-regression-retail-0009\\FP32\\landmarks-regression-retail-0009" -hpe "..\\intel\\head-pose-estimation-adas-0001\\FP32\\head-pose-estimation-adas-0001" -ge "..\\intel\\gaze-estimation-adas-0002\\FP32\\gaze-estimation-adas-0002" -i ..\\bin\\demo.mp4
```

## Documentation
The usage and arguments of main.py are explained below:

```
usage: 
	main.py [-h] -fd FACE_DETECTION_MODEL -hpe HEAD_POSE_ESTIMATION_MODEL
        -fld FACIAL_LANDMARKS_DETECTION_MODEL -ge GAZE_ESTIMATION_MODEL
        -i INPUT [-l CPU_EXTENSION] [-d DEVICE] [-pt PROB_THRESHOLD]]

optional arguments:
	-h, --help            show this help message and exit
	-fd FACE_DETECTION_MODEL, --face_detection_model FACE_DETECTION_MODEL
						Path to a face detection model xml file.
	-hpe HEAD_POSE_ESTIMATION_MODEL, --head_pose_estimation_model HEAD_POSE_ESTIMATION_MODEL
						Path to a head pose estimation model xml file.
	-fld FACIAL_LANDMARKS_DETECTION_MODEL, --facial_landmarks_detection_model FACIAL_LANDMARKS_DETECTION_MODEL
						Path to a facial landmarks detection model xml file.
	-ge GAZE_ESTIMATION_MODEL, --gaze_estimation_model GAZE_ESTIMATION_MODEL
						Path to a gaze estimation model xml file.
	-i INPUT, --input INPUT
						Path to image or video file or enter CAM for using
						webcam.
	-l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
						MKLDNN (CPU)-targeted custom layers.Absolute path to a
						shared library with thekernels impl.
	-d DEVICE, --device DEVICE
						Specify the target device to infer on: CPU, GPU, FPGA
						or MYRIAD is acceptable. Sample will look for a
						suitable plugin for device specified (CPU by default)
	-pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
						Probability threshold for detections filtering(0.5 by
						default)
```


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
