'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore 
from base_model import Model

class Head_Pose_Estimation(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, logger, device, extensions):
        '''
        TODO: Use this to set your instance variables.
        '''
        super().__init__(model_name, logger, device, extensions)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_image = self.preprocess_input(image)
        results = self.exec_network.infer({self.input_blob:processed_image})
        return self.preprocess_output(results)

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
        result = [
            outputs['angle_y_fc'].tolist()[0][0],
            outputs['angle_p_fc'].tolist()[0][0],
            outputs['angle_r_fc'].tolist()[0][0]
        ]
        return result
        #raise NotImplementedError
