'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os  
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore 
from base_model import Model

class Face_Detection(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, logger, device, extensions, prob_threshold=0.6):
        '''
        TODO: Use this to set your instance variables.
        '''
        super().__init__(model_name, logger, device, extensions)
        self.prob_threshold = prob_threshold

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_image = self.preprocess_input(image)
        try:
            results = self.exec_network.infer({self.input_blob:processed_image})
        except Exception as e:
            self.logger.error("{0}".format(e))
            raise ValueError("Inference not done. {0}".format(e))
            
        coordinates = self.preprocess_output(results)
        if (len(coordinates)==0):
            return 0, 0
            
        #coordinates[0] is the first face detected in the frame
        coordinate = coordinates[0] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        coordinate = coordinate.astype(np.int32)        
        detected_part = image[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]
        return detected_part, coordinate
        
        #raise NotImplementedError



    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        processed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        processed_image = np.transpose(np.expand_dims(processed_image, axis=0), (0, 3, 1, 2))
        return processed_image
        #raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coordinates = []
        boxes = outputs[self.output_blob][0][0]
        for box in boxes:
            confidence = box[2]
            if confidence >= self.prob_threshold:
                x_min = box[3]
                y_min = box[4]
                x_max = box[5]
                y_max = box[6]
                coordinates.append([x_min, y_min, x_max, y_max])
        return coordinates
        #raise NotImplementedError
