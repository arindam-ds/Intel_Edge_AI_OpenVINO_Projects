"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network
import logging

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser
    
def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def infer_on_stream(args, client):
    """
    Load the network and parse the SSD output.
    :return: None
    """
    # Flag for the input image
    IMAGE_FLAG = False
    frame_threshold = 10 #number of consecutive frames required for trusted prediction
    frame_buffer = 0
    frame_number = 0
    frame_count_start = 0
    frame_count_end = 0
    last_ppl_count = 0
    total_ppl_count = 0

    # Initialise the class
    infer_network = Network()
    
    # Load the network to IE plugin to get shape of input layer
    infer_network.load_model(args)
    net_input_shape = infer_network.get_input_shape()

    # Checks for live feed
    if args.input == 'CAM':
        input_stream = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        IMAGE_FLAG = True
        input_stream = args.input

    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        print("ERROR! Unable to open video source")
    
    prob_threshold = args.prob_threshold
    initial_w = cap.get(3)
    initial_h = cap.get(4)
    current_frame_detected = False
    last_frame_detected = False
    number_of_same_result_frames = 0
    while cap.isOpened():
        flag, frame = cap.read()
        frame_number+=1
        logging.info("frame_number: {0}".format(frame_number))
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        # Start async inference        
        # Change data layout from HWC to CHW
        
        image = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)  
        
        # Start asynchronous inference for specified request.
        inf_start = time.time()
        tmp_net = infer_network.exec_net(image)
        # Wait for the result
        if infer_network.wait(tmp_net) == 0:
            # Results of the output layer of the network
            result = infer_network.get_output()
            det_time = time.time() - inf_start
            current_count = 0
            for obj in result[0][0]:
                # Draw bounding box for object when it's probability is more than
                #  the specified threshold
                if obj[2] > prob_threshold:
                    xmin = int(obj[3] * initial_w)
                    ymin = int(obj[4] * initial_h)
                    xmax = int(obj[5] * initial_w)
                    ymax = int(obj[6] * initial_h)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
                    current_count = current_count + 1
            
            if current_count > last_ppl_count or current_count > 0:
                current_frame_detected = True
            else:
                current_frame_detected = False
                
    
            inf_time_message = "Inference time: {0:.3f}ms, Total Count: {1}".format(det_time * 1000, total_ppl_count)
            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            # If new person enters the video
            logging.info("last_frame_detected: {0}".format(last_frame_detected))
            logging.info("current_frame_detected: {0}".format(current_frame_detected))
            logging.info("============================================================")
            
            if last_frame_detected == False and current_frame_detected == True:
                if frame_buffer == 0:
                    frame_count_start = frame_number
                    total_ppl_count = total_ppl_count + current_count - last_ppl_count
                else:
                    frame_buffer = 0
            
            elif last_frame_detected == True and current_frame_detected == False:
                frame_buffer+=1
            
            elif last_frame_detected == False and current_frame_detected == False:
                if frame_buffer > 0:
                    frame_buffer+=1
                    
                if frame_buffer==frame_threshold:
                    frame_count_end = frame_number   
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frame = frame_count_end - frame_buffer - frame_count_start + 1
                    duration = round(total_frame/ fps)
                    logging.info("frame_buffer: {0}".format(frame_buffer))
                    logging.info("frame_count_start: {0}".format(frame_count_start))
                    logging.info("frame_count_end: {0}".format(frame_count_end))
                    logging.info("total_frame: {0}".format(total_frame))
                    logging.info("duration: {0}".format(duration))
                    logging.info("total_ppl_count: {0}".format(total_ppl_count))
                    
                    client.publish("person/duration", json.dumps({"duration": duration}))
                    client.publish("person", json.dumps({"total": total_ppl_count}))
                
                    frame_buffer = frame_count_end = frame_count_start = 0
            else:
                if frame_buffer > 0:
                    frame_buffer+=1
                
            client.publish("person", json.dumps({"count": current_count}))
            last_ppl_count = current_count

            if key_pressed == 27:
                break

        # Send frame to the ffmpeg server
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        if IMAGE_FLAG:
            cv2.imwrite('output_image.jpg', frame)
            
        last_frame_detected = current_frame_detected
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()

def main():
    # Grab command line args
    logging.basicConfig(filename='app.log', format='%(asctime)s - %(message)s', level=logging.INFO)
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()   
    # Perform inference on the input stream
    infer_on_stream(args, client)
    logging.close()
    
if __name__ == '__main__':
    main()
    exit(0)