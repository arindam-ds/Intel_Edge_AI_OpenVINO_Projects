'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class Gaze_Estimation:
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
        try:
            self.net = self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Please check the model path.")
        
        ### TODO: Check for supported layers ###
        checked = self.check_model()
        if not checked:
            sys.exit(1)
        
        self.exec_network = self.core.load_network(self.net, args.device)
        
        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))
        self.input_shape = self.net.inputs[self.input_blob].shape
        self.output_shape = self.net.outputs[self.output_blob].shape        
        return
        #raise NotImplementedError

    def predict(self, left_eye_image, right_eye_image, head_pose_estimation_val):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye_processed, right_eye_processed = self.preprocess_input(left_eye_image,right_eye_image)
        outputs = self.exec_net.infer({'left_eye_image':left_eye_processed, 'right_eye_image':right_eye_processed, 'head_pose_angles':head_pose_estimation_val})
        mouse_coordinates, gaze_val = self.preprocess_output(outputs, head_pose_estimation_val)
        return mouse_coordinates, gaze_val
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

    def preprocess_input(self, left_eye_image, right_eye_image):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        left_eye_image = cv2.resize(left_eye_image, (self.input_shape[3], self.input_shape[2]))
        right_eye_image = cv2.resize(right_eye_image, (self.input_shape[3], self.input_shape[2]))
        
        left_eye_processed = np.transpose(np.expand_dims(left_eye_image, axis=0), (0, 3, 1, 2))
        right_eye_processed = np.transpose(np.expand_dims(right_eye_image, axis=0), (0, 3, 1, 2))
        return left_eye_processed, right_eye_processed
        #raise NotImplementedError

    def preprocess_output(self, outputs, head_pose_estimation_val):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        roll_value = head_pose_estimation_output[2]
        outputs = outputs[self.output_blob].tolist()[0]
        
        cos_val = math.cos(roll_value * math.pi / 180)
        sin_val = math.sin(roll_value * math.pi / 180)

        x_value = outputs[0] * cos_val + outputs[1] * sin_val
        y_value = outputs[1] * cos_val - outputs[0] * sin_val

        return (x_value, y_value), outputs
        #raise NotImplementedError
