import cv2
import os
import numpy as np
import logging
from face_detection import Face_Detection
from head_pose_estimation import Head_Pose_Estimation
from facial_landmarks_detection import Facial_Landmarks_Detection
from gaze_estimation import Gaze_Estimation
from mouse_controller import MouseController
from input_feeder import InputFeeder
from argparse import ArgumentParser
import time

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help="Path to a face detection model xml file.")
    parser.add_argument("-hpe", "--head_pose_estimation_model", required=True, type=str,
                        help="Path to a head pose estimation model xml file.")
    parser.add_argument("-fld", "--facial_landmarks_detection_model", required=True, type=str,
                        help="Path to a facial landmarks detection model xml file.")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
                        help="Path to a gaze estimation model xml file.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or enter CAM for using webcam.")
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
    parser.add_argument("-flag", "--visualization_flag", required=False, nargs='+',
                        default=[],
                        help="Example: --flag fd hpe ge fld (Seperate each flag by space)"
                             "for getting the visualization of different model outputs of each frame."
                             "fd for Face Detection Model, hpe for Head Pose Estimation Model,"
                             "fld for Facial Landmark Detection Model, ge for Gaze Estimation Model.")
    return parser

def main():
    logging.basicConfig(filename='app.log', format='%(asctime)s - %(message)s', level=logging.INFO)
    args = build_argparser().parse_args()
    input_file = args.input
    visualization_flags = args.visualization_flag

    if input_file.lower()=="cam":
        feed = InputFeeder("cam")
    elif not os.path.isfile(input_file):
            logging.error("Given input file is not found.")
            exit(1)
    else:
        feed = InputFeeder("video", input_file)
    
    logging.info("gaze_estimation_model Model: {0}".format(args.gaze_estimation_model))
    start_model_load_time=time.time()
    
    face_detection = Face_Detection(args.face_detection_model, logging, args.device, args.cpu_extension, args.prob_threshold)
    head_pose_estimation = Head_Pose_Estimation(args.head_pose_estimation_model, logging, args.device, args.cpu_extension)
    facial_landmarks_detection = Facial_Landmarks_Detection(args.facial_landmarks_detection_model, logging, args.device, args.cpu_extension)
    gaze_estimation = Gaze_Estimation(args.gaze_estimation_model, logging, args.device, args.cpu_extension)
    
    
    face_detection.load_model()
    head_pose_estimation.load_model()
    facial_landmarks_detection.load_model()
    gaze_estimation.load_model()
    total_model_load_time = time.time() - start_model_load_time
    
    mouse_controller = MouseController("medium", "fast")
    logging.info("Face Detection model loaded.")
    logging.info("Head Pose Estimation model loaded.")
    logging.info("Facial Landmarks Detection model loaded.")
    logging.info("Gaze Estimation model loaded.")
    
    feed.load_data()
    frame_count = 0
    start_inference_time=time.time()
    
    for frame in feed.next_batch():
        try:
            frame_count+=1
            if frame_count%5 == 0:
                cv2.imshow('video',cv2.resize(frame,(500,500)))
            key = cv2.waitKey(60)
            
            #inference from Face_Detection model.
            detected_part, coordinate = face_detection.predict(frame.copy())
            if type(detected_part) == int:
                logging.info("No face detected in frame no. {0}.".format(frame_count))
                if key==27:
                    break
                continue
            
            #inference from Head_Pose_Estimation model.
            hpe_result = head_pose_estimation.predict(detected_part.copy())
            
            #inference from Facial_Landmarks_Detection model.
            left_eye, right_eye, eye_coordinates = facial_landmarks_detection.predict(detected_part.copy())
            
            #inference from Gaze_Estimation model.
            mouse_coordinates, gaze_val = gaze_estimation.predict(left_eye, right_eye, hpe_result)
            
            #cv2.imshow("visualization",cv2.resize(frame,(500,500)))
            
            if frame_count%5 == 0:
                mouse_controller.move(mouse_coordinates[0], mouse_coordinates[1])    
            if key==27:
                break
        except Exception as e:
                break
    
    total_time = time.time()-start_inference_time
    total_inference_time = round(total_time, 1)
    fps = frame_count/total_inference_time
    logging.info("total_model_load_time: {}".format(total_model_load_time))
    logging.info("total_inference_time: {}".format(total_inference_time))
    logging.info("fps: {}".format(fps))
    
    logging.info("video finished.")
    cv2.destroyAllWindows()   
    feed.close()    
    #logging.close()

if __name__ == '__main__':
    main()
    exit(0)