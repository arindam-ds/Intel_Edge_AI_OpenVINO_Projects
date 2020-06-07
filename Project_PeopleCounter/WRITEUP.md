# Project Write-Up

In this write-up we will discuss about the OpenVINO™ Toolkit and its impact on performance, as well as articulating the use cases of this People Counter application deployed at the edge. 

## Explaining Custom Layers

Neural Networks (NN) contain different layers stacked together. NN can be implemented using different frameworks such as Tensorflow, Pytorch, Caffe, MXNet etc. Most of the NN layers are supported by OpenVino and these are called known layer. There are many NN layers of different frameworks which are not supported natively by OpenVino framework and hence these are not listed as known layers. If the model’s topology contains any such layer, Model Optimizer considers it as custom layer which is not supported and throws an error. 

The process behind converting custom layers involves registering that layer as extensions to the Model Optimizer. Then Model Optimizer generates a valid and optimized Intermediate Representation.

The potential reason for handling custom layers are to convert models having unsupported layers into intermediate representations.

## Comparing Model Performance

I tried with three different models:

- ssd_inception_v2_coco_2018_01_28
- ssd_mobilenet_v2_coco_2018_03_29
- ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03

These are pretrained models available in [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

The comparison on size is given below:

| |SSD Inception V2 COCO|SSD Mobilenet V2 COCO|SSD Resnet V1|
|-|-|-|-|
|Before Conversion (in MB)|100|68|131|
|After Conversion (in MB)|98|66|330|

I tried to use the models in a Jupyter Notebook from it’s frozen form and without using OpenVino to measure their performance. Then, after conversion into IR form, we only need the xml and bin files. The inference time of the three models are shown below:

| |SSD Inception V2 COCO|SSD Mobilenet V2 COCO|SSD Resnet V1|
|-|-|-|-|
|Before Conversion (in ms)|75|78|2040|
|After Conversion (in ms)|60|58|1200|

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
i)	Counting total footprint in an event or gathering
ii)	Counting daily customer visit in a shopping mall
iii)Counting average amount of time spent by a person at some point of interest/advertisement/display etc.

Each of these use cases would be useful because these will directly help business for crucial decision-making and thus can increase their revenue. 

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:
- For this edge application to be performing efficiently there should be a good source of light present. Lightning improves visibility and well visible objects/people would be identified by the app with more confidence. 

- Camera focal length also matters a lot. The focal length of a camera lens is the distance between the lens and the image sensor when the subject is in focus. Focal length controls the angle of view and magnification of a photograph. A properly focused camera can send good quality image or video stream resulting into good quality prediction by the app.

- The edge deployed models used here have less inference time along with pretty good accuracy. This model can predict people from frame with high confidence (> 80% precision). It cuts the latency overhead for detection from video streams due to its less inference time. Before the inference, video frame is resized according to the input shape of the model. 

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: ssd_inception_v2_coco_2018_01_28
  - This model belongs to Tensoflow’s pretrained model zoo.
  - I converted the model to an Intermediate Representation with the following arguments:
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```
  - The model performed well but sometimes missing the presence of people in the frame.
  - I tried to improve the model for the app by converting into IR form. And then by changing the probability threshold argument.
  
- Model 2: ssd_mobilenet_v2_coco_2018_03_29
  - This model belongs to Tensoflow’s pretrained model zoo.
  - I converted the model to an Intermediate Representation with the following arguments:
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

  - The model is better than the first one in terms of size and inference time. It misses the presence of people in the frame for lesser number of times.
  - I tried to improve the model for the app by converting into IR form. And then by changing the probability threshold argument.

- Model 3: ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
  - This model belongs to Tensoflow’s pretrained model zoo.
  - I converted the model to an Intermediate Representation with the following arguments:
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ./frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
```
  - This model’s performance is the worst among three.
  - This is not up to mark in terms of size and infrence time too.
  - Due to it's size, model loading time and inference time are high. Precision is low. Not suitable for edge application.
  
## OpenVino Model

Finally I tried with OpenVino pretrained model [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html). This is based on the MobileNetV2-like backbone. This model performed well for detecting people in terms of precision, inference time and model size. 
  - Model's size: 3.1 MB
  - Model's inference time: 48 ms
  
OpenVino pretrained model **person-detection-retail-0013** is the best model for this Edge App.