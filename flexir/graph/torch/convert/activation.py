
from .common import *

def convert_torch_nn_functional_relu(func=F.relu, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_relu_arguments(input, inplace=False):
        return { 'input': input }
    # ---
    args_dict = get_torch_nn_functional_relu_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... ReLU')
    return convert_demonstrate(func, inputs, output)


def convert_Module_ReLU(module=nn.ReLU(), inputs=None, output=None):
    logger.debug('Converting... ReLU')
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_hswish(func=F.hardswish, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_hswish_arguments(input, inplace=False):
        return { 'input': input }
    # ---
    args_dict = get_torch_nn_functional_hswish_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... hswish')
    return convert_demonstrate(func, inputs, output)


def convert_Module_HSwish(module=None, inputs=None, output=None):
    logger.debug('Converting... hswish')
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_hsigmoid(func=F.hardsigmoid, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_hsigmoid_arguments(input, inplace=False):
        return { 'input': input }
    # ---
    args_dict = get_torch_nn_functional_hsigmoid_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... hsigmoid')
    return convert_demonstrate(func, inputs, output)


def convert_Module_HSigmoid(module=None, inputs=None, output=None):
    logger.debug('Converting... hsigmoid')
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_relu6(func=F.relu6, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_relu6_arguments(input, inplace=False):
        return { 'input': input }
    # ---
    args_dict = get_torch_nn_functional_relu6_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... ReLU6')
    return convert_demonstrate(func, inputs, output)


def convert_Module_ReLU6(module=nn.ReLU6(), inputs=None, output=None):
    logger.debug('Converting... ReLU6')
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_leaky_relu(func=F.leaky_relu, inputs=None, output=None, *args, **kwargs):
    def get_torch_nn_functional_leaky_relu_arguments(input=None, negative_slope=0.01, inplace=False):
        return { 'input':input, 'inplace': inplace, 'negative_slope': negative_slope }
    args_dict = get_torch_nn_functional_leaky_relu_arguments(*args, **kwargs)
    logger.debug('Converting... leaky ReLU, alpha {}',
        args_dict['negative_slope'],
    )
    return convert_demonstrate(func, inputs, output)


def convert_Module_LeakyReLU(module=nn.LeakyReLU(negative_slope=0.1), inputs=None, output=None):
    logger.debug('Converting... leaky ReLU, alpha {}',
        module.negative_slope,
    )
    return convert_demonstrate(module, inputs, output)


def convert_torch_sigmoid(func=torch.sigmoid, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_sigmoid_arguments(input, out=None):
        return { 'input': input, 'out':out }
    # ---
    args_dict = get_torch_sigmoid_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... sigmoid')
    return convert_demonstrate(func, inputs, output)


def convert_torch_nn_functional_sigmoid(func=F.sigmoid, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_sigmoid_arguments(input, inplace=False):
        return { 'input': input }
    # ---
    args_dict = get_torch_nn_functional_sigmoid_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... sigmoid')
    return convert_demonstrate(func, inputs, output)

def convert_Module_Sigmoid(module=nn.Sigmoid(), inputs=None, output=None):
    logger.debug('Converting... sigmoid')
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_tanh(func=F.tanh, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_tanh_arguments(input, inplace=False):
        return { 'input': input }
    # ---
    args_dict = get_torch_nn_functional_tanh_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... tanh')
    return convert_demonstrate(func, inputs, output)


def convert_Module_Tanh(module=nn.Tanh(), inputs=None, output=None):
    logger.debug('Converting... tanh')
    return convert_demonstrate(module, inputs, output)


def convert_torch_tanh(module=torch.tanh, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_tanh_arguments(input):
        return { 'input': input }
    # ---
    args_dict = get_torch_nn_functional_tanh_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... tanh')
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_prelu(func=F.prelu, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_prelu_arguments(input, weight, inplace=False):
        return {'input': input}, {'weight': weight}
    # ---
    args_dict = get_torch_nn_functional_prelu_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... pReLU')
    weight = args_dict['weight']
    return convert_demonstrate(func, inputs, output)


def convert_Module_PReLU(module=nn.PReLU(), inputs=None, output=None):
    logger.debug('Converting... pReLU')
    sd = module.state_dict()
    weight = sd['weight']
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_elu(func=F.elu, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_elu_arguments(input, alpha=1.0, inplace=False):
        return {'input': input, 'alpha': alpha}
    # ---
    args_dict = get_torch_nn_functional_elu_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... ELU, alpha {}',
        args_dict['alpha']
    )
    return convert_demonstrate(func, inputs, output)


def convert_Module_ELU(module=nn.ELU(), inputs=None, output=None):
    logger.debug('Converting... ELU, alpha {}',
        module.alpha
    )
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_silu(func=F.silu, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_silu_arguments(input, inplace=False):
        return {'input': input}
    # ---
    args_dict = get_torch_nn_functional_silu_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... SiLU')
    return convert_demonstrate(func, inputs, output)


def convert_Module_SiLU(module=nn.SiLU(), inputs=None, output=None):
    logger.debug('Converting... SiLU')
    return convert_demonstrate(module, inputs, output)


def convert_Module_ReLUXPerChannel(module=None, inputs=None, output=None):
    logger.debug('Converting... ReLUXPerChannel')
    sd = module.state_dict()
    weight = sd['scale']
    return convert_demonstrate(module, inputs, output)

