from openvino.inference_engine import IENetwork, IECore 

class Model:
    def __init__(self, model_name, logger, device='CPU', extensions=None):
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
        self.logger = logger

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        if self.extensions and 'CPU' in self.device:
            self.core.add_extension(self.extensions, 'CPU')
        try:
            self.net = IENetwork(model=self.model_structure, weights=self.model_weights)
            self.logger.info("{} files are read successfully.".format(self.model_structure.split(".")[0]))
        except Exception as e:
            self.logger.error("Could not Initialise the {0} network. {1}".format(self.model_structure.split(".")[0], format(e)))
            raise ValueError("Could not Initialise the network. {}".format(e))
        
        ### Check for supported layers ###
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
    
    def check_model(self):
        supported_layers = self.core.query_network(network=self.net, device_name=self.device)        
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            self.logger.error("Following layers are not supported by \
                          the core for specified device {}:\n {}".
                          format(self.core.device,\
                                 ', '.join(unsupported_layers)))
            self.logger.error("Please try to specify cpu extensions library path"
                          " in command line parameters using -l "
                          "or --cpu_extension command line argument")
            return False
        else:
            return True