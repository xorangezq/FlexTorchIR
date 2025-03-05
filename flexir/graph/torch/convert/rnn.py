
from .common import *

def convert_Module_RNNCell(module=nn.RNNCell(1, 1), inputs=None, output=None):
    if module.nonlinearity == 'tanh':
        # info._activation_type = ENUM_TYPE_ACTIVATION['TanH']
        # info._activation_alpha = 1.0
        # info._activation_beta = 0.0
        pass
    elif module.nonlinearity == 'relu':
        # info._activation_type = ENUM_TYPE_ACTIVATION['ReLU']
        # info._activation_alpha = 1.0
        # info._activation_beta = 0.0
        pass
    else:
        raise NotImplementedError

    sd = module.state_dict()
    return convert_demonstrate(module, inputs, output)


def convert_Module_RNN(module=nn.RNN(1, 1, 1), inputs=None, output=None):

    if module.mode == 'RNN_TANH':
        # info._activation_type = ENUM_TYPE_ACTIVATION['TanH']
        # info._activation_alpha = 1.0
        # info._activation_beta = 0.0
        return convert_demonstrate(module, inputs, output)
    elif module.mode == 'RNN_RELU':
        # info._activation_type = ENUM_TYPE_ACTIVATION['ReLU']
        # info._activation_alpha = 1.0
        # info._activation_beta = 0.0
        return convert_demonstrate(module, inputs, output)
    else:
        raise NotImplementedError('Unknown module.mode = {}'.format(module.mode))

    sd = module.state_dict()
    return convert_demonstrate(module, inputs, output)


def convert_Module_LSTMCell(module=nn.LSTMCell(1,1,1), inputs=None, output=None):
    sd = module.state_dict()
    return convert_demonstrate(module, inputs, output)


def convert_Module_LSTM(module=nn.LSTM(1,1,1), inputs=None, output=None):
    sd = module.state_dict()
    return convert_demonstrate(module, inputs, output)


def convert_Module_GRUCell(module=nn.GRUCell(1,1,1), inputs=None, output=None):
    sd = module.state_dict()
    return convert_demonstrate(module, inputs, output)


def convert_Module_GRU(module=nn.GRU(1,1,1), inputs=None, output=None):
    sd = module.state_dict()
    return convert_demonstrate(module, inputs, output)
