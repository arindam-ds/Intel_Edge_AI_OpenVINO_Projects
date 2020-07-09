'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore 

class Head_Pose_Estimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        self.extensions = extensions
        self.core = None
        self.net = None
        self.exec_network = None
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None
        
        #raise NotImplementedError

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        ######
        if self.extensions and 'CPU' in self.device:
            self.core.add_extension(self.extensions, 'CPU')
        ######
        try:
            ######self.net = self.core.read_network(model=self.model_structure, weights=self.model_weights)
            self.net = IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. {}".format(e))
        
        ### TODO: Check for supported layers ###
        checked = self.check_model()
        if not checked:
            sys.exit(1)
        
        self.exec_network = self.core.load_network(self.net, self.device)
        
        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))
        self.input_shape = self.net.inputs[self.input_blob].shape
        self.output_shape = self.net.outputs[self.output_blob].shape        
        return
        #raise NotImplementedError

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_image = self.preprocess_input(image)
        #results = self.exec_network.infer(inputs={self.input_blob : processed_image})
        results = self.exec_network.infer({self.input_blob:processed_image})
        return self.preprocess_output(results)
        #raise NotImplementedError

    def check_model(self):
        supported_layers = self.core.query_network(network=self.net, device_name=self.device)        
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Following layers are not supported by "
                          "the core for specified device {}:\n {}".
                          format(self.core.device,
                                 ', '.join(unsupported_layers)))
            print("Please try to specify cpu extensions library path"
                          " in command line parameters using -l "
                          "or --cpu_extension command line argument")
            return False
        else:
            return True
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
        result = [
            outputs['angle_y_fc'].tolist()[0][0],
            outputs['angle_p_fc'].tolist()[0][0],
            outputs['angle_r_fc'].tolist()[0][0]
        ]
        return result
        #raise NotImplementedError
