# Computer Pointer Controller

In this project we have developed an Intel OpenVINO-based Edge-AI application that can control and move the pointer on computer screen based on the movement of gaze of human eye. This application can take an image, video or webcam feed as input. From the video input, it first detects the presence of face in the input frame. Then it detects the landmark of the face and pose of the head. Finally from the detected landmarks and pose, it estimates the gaze of the eyes. Based on this gaze, computer pointer is controlled. The pipeline flow of the models ar shown below:

![Pipeline flow](https://github.com/arin1405/Intel_Edge_AI_OpenVINO_Projects/blob/master/Computer%20Pointer%20Controller/pipeline.png "Pipeline flow")

## Project Set Up and Installation
For successfull development and execution of the project, Intel [OpenVINO](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) needs to be installed. The installation document can be found [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html). I have used OpenVINO 2020.1.033 version for this project.

We need four pretrained models of Intel OpenVINO. These models can be downloaded by model downloader script `downloader.py`.

**1. Face Detection Model**
```
python C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.1.033\\deployment_tools\\tools\\model_downloader\\downloader.py --name "face-detection-adas-binary-0001"
```
**2. Facial Landmarks Detection Model**
```
python C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.1.033\\deployment_tools\\tools\\model_downloader\\downloader.py --name "landmarks-regression-retail-0009"
```
**3. Head Pose Estimation Model**
```
python C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.1.033\\deployment_tools\\tools\\model_downloader\\downloader.py --name "head-pose-estimation-adas-0001"
```
**4. Gaze Estimation Model**
```
python C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.1.033\\deployment_tools\\tools\\model_downloader\\downloader.py --name "gaze-estimation-adas-0002"
```

## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

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
