import os
import yaml
import numpy as np


class Parameters:
    """
    The parameters class, crucial to read configuration files and run a complex code

    Keyword arguments:
    input file -- path to the .yaml configuration file where daneel will extract all the important parameters
    Return: a Python dictionary with all the parameters contained in the input file
    """

    def __init__(self, input_file):
        if os.path.exists(input_file) and os.path.isfile(input_file):
            with open(input_file) as in_f:
                self.params = yaml.load(in_f, Loader=yaml.FullLoader)

        for par in list(self.params.keys()):
            if self.params[par] == "None":
                self.params[par] = None

    def get(self, param):
        return self.params[param]
