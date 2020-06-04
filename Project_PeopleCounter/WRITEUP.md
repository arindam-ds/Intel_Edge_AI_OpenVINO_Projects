# Project Write-Up

In this write-up we will discuss about the OpenVINO™ Toolkit and its impact on performance, as well as articulating the use cases of this People Counter application deployed at the edge. 

## Explaining Custom Layers

Neural Networks (NN) contain different layers stacked together. NN can be implemented using different frameworks such as Tensorflow, Pytorch, Caffe, MXNet etc. Most of the NN layers are supported by OpenVino and these are called known layer. There are many NN layers of different frameworks which are not supported natively by OpenVino framework and hence these are not listed as known layers. If the model’s topology contains any such layer, Model Optimizer considers it as custom layer which is not supported and throws an error. 

The process behind converting custom layers involves registering that layer as extensions to the Model Optimizer. Then Model Optimizer generates a valid and optimized Intermediate Representation.

The potential reason for handling custom layers are to convert models having unsupported layers into intermediate representations.

## Comparing Model Performance

I tried with different models. I tried to use the model in a Jupyter Notebook from it’s frozen form and without using OpenVino. The model "ssd_mobilenet_v2_coco_2018_03_29" is having size of 201 MB. It consists of frozen model in protobuf format, model.ckpt.data-00000-of-00001, model.ckpt.index, model.ckpt.meta, saved_model etc. After conversion into IR form, we only need the xml and bin files. These are having size of 264 KB and 65 MB. The inference time of the model pre-conversion was 65 ms whereas after conversion it was 30 ms.

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

- Model 1: person-detection-retail-0013
  - This is Intel OpenVino pre-trained model. I did not have to convert it into IR form.
  - The model performed pretty good in predicting the person’s presence in a frame.
  - I tried to improve the model for the app by changing the probability thresholds while running this.
  
- Model 2: ssd_mobilenet_v2_coco_2018_03_29
  - This model belongs to Tensoflow’s pretrained model zoo.
  - I converted the model to an Intermediate Representation with the following arguments:
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

  - The model performed well but sometimes with lesser precision than the first model.
  - I tried to improve the model for the app by converting into IR form. And then by changing the probability threshold argument.

- Model 3: faster_rcnn_inception_v2_coco_2018_01_28
  - This model belongs to Tensoflow’s pretrained model zoo.
  - I converted the model to an Intermediate Representation with the following arguments:
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ./frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
```
  - The model’s precision was better than the second model.
