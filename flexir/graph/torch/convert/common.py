
# pytorch imports
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable, Function
import torch.nn.functional as F

# other imports
import numpy as np
import math
import base64
import copy

from flexir.graph.layers import LayerDemonstrate, LayerPlaceholder
from flexir.utilities.logger import logger, ASSERT, TENSOR_PEEK


# <Function>: Compute weight_v 2 norm
def norm_except_dim(v, pow, dim):
    if dim == -1:
        return torch.norm(v, pow)
    elif dim == 0:
        output_size = (v.shape[0],) + (1,) * (v.ndim - 1)
        return torch.norm(v, pow, (1, 2)).view(output_size)
    elif dim == (v.ndim - 1):
        output_size = (1,) * (v.ndim - 1) + (v.shape[v.ndim - 1])
        return torch.norm(v.view((-1, v.shape[v.ndim - 1])), pow, 0).view(output_size)
    else:
        return norm_except_dim(v.swapaxes(0, dim), pow, dim).swapaxes(0,dim)

# <Function>: Normalized weight restored to original weight
def restore_weighted_norm_inplace(weight_g, weight_v):
    tmp = norm_except_dim(weight_v, 2, 0)
    original_weight = weight_v * (weight_g / tmp)
    return original_weight


def recurrent_split_arguments(arguments=[1, [3, [10, 8], [5, 6, 7]]]):
    args = []
    if isinstance(arguments, tuple) or isinstance(arguments, list):
        for arg in arguments:
            args = args + recurrent_split_arguments(arg)
    else:
        args.append(arguments)

    return args


class AsymmetricPaddingWrapper(nn.Module):
    def __init__(self, module, paddings_lrtb):
        super(AsymmetricPaddingWrapper, self).__init__()
        self.pad = nn.ZeroPad2d(paddings_lrtb)
        self.core_module = module
    def forward(self, *inps, **kwinps):
        y = self.pad(*inps)
        y = self.core_module(y)
        return y


def convert_torch_unrecognized(opdesc, inputs, outputs, *args, **kwargs):
    map_shapes = lambda tensor: [i for i in tensor.shape]
    map_list_of_shapes = lambda tensors: [map_shapes(t) for t in tensors]

    assert(isinstance(inputs, (tuple, list)))
    assert(not isinstance(outputs, (tuple, list)))

    info = LayerPlaceholder()
    info._input_ids = [str(id(t)) for t in inputs] if isinstance(inputs, list) else [str(id(inputs))]
    info._input_shapes = map_list_of_shapes(inputs)
    info._output_shape = map_shapes(outputs)
    info._desc = 'Op: {opdesc}'.format(opdesc=opdesc)
    return info


def convert_demonstrate(opdesc, inputs, outputs, *args, **kwargs):
    map_shapes = lambda tensor: [i for i in tensor.shape]
    map_list_of_shapes = lambda tensors: [map_shapes(t) for t in tensors]

    assert(isinstance(inputs, (tuple, list)))
    assert(not isinstance(outputs, (tuple, list)))

    info = LayerDemonstrate()
    info._input_ids = [str(id(t)) for t in inputs] if isinstance(inputs, list) else [str(id(inputs))]
    info._input_shapes = map_list_of_shapes(inputs)
    info._output_shape = map_shapes(outputs)
    info._desc = 'Op: {opdesc}'.format(opdesc=opdesc)
    return info


def convert_Module_Identity(module=nn.Identity(), inputs=None, output=None):
    logger.debug('Layer - Identity')
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_softmax(func=F.softmax, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_softmax_arguments(input, dim=None, _stacklevel=3, dtype=None):
        return { 'input': input, 'dim': dim }
    # ---
    args_dict = get_torch_nn_functional_softmax_arguments(*args, **kwargs)
    # ---
    logger.debug('Layer - Softmax, dim{}', args_dict['dim'])
    return convert_demonstrate(func, inputs, output)

def convert_Module_Softmax(mod=nn.Softmax(dim=1), inputs=None, output=None):
    logger.debug('Layer - Softmax, dim{}', mod.dim)
    return convert_demonstrate(func, inputs, output)


def convert_torch_nn_functional_pad(func=F.pad, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_pad_arguments(input, pad, mode='constant', value=0):
        return {'input': input, 'pad': pad, 'mode': mode, 'value': value}
    # ---
    args = get_torch_nn_functional_pad_arguments(*args, **kwargs)
    # ---
    if args['mode'] == 'constant':
        if len(args['pad']) == 2:
            logger.debug('Layer - Pad1d, pad {} {}, Const {}',
                args['pad'][0],
                args['pad'][1],
                args['value'],
            )
        elif len(args['pad']) == 4:
            logger.debug('Layer - Pad2d, pad {} {} {} {}, Const {}',
                args['pad'][0],
                args['pad'][1],
                args['pad'][2],
                args['pad'][3],
                args['value'],
            )
        else:
            raise NotImplementedError
    elif args['mode'] == 'reflect':
        if len(args['pad']) == 2:
            logger.debug('Layer - Pad, pad {} {}, Reflect',
                args['pad'][0],
                args['pad'][1],
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return convert_demonstrate(func, inputs, output)


def convert_Module_ZeroPad2d(module=nn.ZeroPad2d((0, 0, 0, 0)), inputs=None, output=None):
    logger.debug('Layer - Pad2d, pad {} {} {} {}, Const 0',
        module.padding[0],
        module.padding[1],
        module.padding[2],
        module.padding[3],
    )
    return convert_demonstrate(module, inputs, output)


def convert_Module_ReflectionPad1d(module=nn.ReflectionPad1d((0, 0)), inputs=None, output=None):
    logger.debug('Layer - Pad1d, pad {} {} {} {}, Reflect',
        module.padding[0],
        module.padding[1],
    )
    return convert_demonstrate(module, inputs, output)


def convert_Module_Conv1d(module=nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'), inputs=None, output=None):
    logger.debug('Converting... Conv1d, C{} N{} K{} S{} P{}{} D{}x{} G{} {}',
        module.in_channels,
        module.out_channels,
        module.kernel_size[0] if isinstance(module.kernel_size, (list, tuple)) else module.kernel_size,
        module.stride[0] if isinstance(module.stride, (list, tuple)) else module.stride,
        module.padding[0] if isinstance(module.padding, (list, tuple)) else module.padding,
        module.padding[1] if isinstance(module.padding, (list, tuple)) else module.padding,
        module.dilation[0] if isinstance(module.dilation, (list, tuple)) else module.dilation,
        module.groups,
        'hasBias' if module.bias is not None else '',
    )
    sd = module.state_dict()
    if 'weight' not in sd.keys() and 'weight_g' in sd.keys():
        weight = restore_weighted_norm_inplace(sd['weight_g'], sd['weight_v'])
    else:
        weight = sd['weight']
    bias = sd['bias']
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_conv2d(func=F.conv2d, inputs=None, output=None, *args, **kwargs):
    def get_torch_nn_functional_conv2d_arguments(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return input , { 'stride':stride, 'padding':padding, 'dilation':dilation, 'groups':groups }, { 'weight':weight, 'bias':bias }
    inp, args_dict, params_dict = get_torch_nn_functional_conv2d_arguments(*args, **kwargs)
    logger.debug('Converting... Conv2d, C{} N{} K{}x{} S{}x{} P{}{} D{}x{} G{} {}',
        params_dict['weight'].size(1) * args_dict['groups'],
        params_dict['weight'].size(0),
        params_dict['weight'].size(2),
        params_dict['weight'].size(3),
        args_dict['stride'][0] if isinstance(args_dict['stride'], (list, tuple)) else args_dict['stride'],
        args_dict['stride'][1] if isinstance(args_dict['stride'], (list, tuple)) else args_dict['stride'],
        args_dict['padding'][0] if isinstance(args_dict['padding'], (list, tuple)) else args_dict['padding'],
        args_dict['padding'][1] if isinstance(args_dict['padding'], (list, tuple)) else args_dict['padding'],
        args_dict['dilation'][0] if isinstance(args_dict['dilation'], (list, tuple)) else args_dict['dilation'],
        args_dict['dilation'][1] if isinstance(args_dict['dilation'], (list, tuple)) else args_dict['dilation'],
        args_dict['groups'],
        'hasBias' if args_dict['bias'] else '',
    )
    weight = params_dict['weight']
    bias = params_dict['bias'] if args_dict['bias'] else None
    return convert_demonstrate(func, inputs, output)


def convert_Module_Conv2d(module=nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=2, bias=True, padding_mode='zeros'), inputs=None, output=None):
    logger.debug('Converting... Conv2d, C{} N{} K{}x{} S{}x{} P{}{} D{}x{} G{} {}',
        module.in_channels,
        module.out_channels,
        module.kernel_size[0],
        module.kernel_size[1],
        module.stride[0],
        module.stride[1],
        module.padding[0],
        module.padding[1],
        module.dilation[0],
        module.dilation[1],
        module.groups,
        'hasBias' if module.bias is not None else '',
    )
    sd = module.state_dict()
    weight = sd['weight']
    bias = sd['bias'] if module.bias is not None else None
    return convert_demonstrate(module, inputs, output)


def convert_Module_ConvTranspose2d(module=nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=2, bias=True, padding_mode='zeros'), inputs=None, output=None):
    logger.debug('Converting... ConvTranspose2d, C{} N{} K{}x{} S{}x{} P{}{}{}{} D{}x{} G{} {}',
        module.in_channels,
        module.out_channels,
        module.kernel_size[0],
        module.kernel_size[1],
        module.stride[0],
        module.stride[1],
        module.dilation[0]*(module.kernel_size[0]-1)+1-module.padding[0]-1,
        module.dilation[1]*(module.kernel_size[1]-1)+1-module.padding[1]-1,
        module.dilation[0]*(module.kernel_size[0]-1)+1-module.stride[0]-module.padding[0]+module.output_padding[0],
        module.dilation[1]*(module.kernel_size[1]-1)+1-module.stride[1]-module.padding[1]+module.output_padding[1],
        module.dilation[0],
        module.dilation[1],
        module.groups,
        'hasBias' if module.bias is not None else '',
    )
    sd = module.state_dict()
    weight = sd['weight']
    bias = sd['bias'] if module.bias is not None else None
    return convert_demonstrate(module, inputs, output)


def convert_Module_ConvTranspose1d(module=nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=2, bias=True, padding_mode='zeros'), inputs=None, output=None):
    logger.debug('Converting... ConvTranspose1d, C{} N{} K{} S{} P{} NP{} D{} G{} {}',
        module.in_channels,
        module.out_channels,
        module.kernel_size[0] if isinstance(module.kernel_size, (list, tuple)) else module.kernel_size,
        module.stride[0] if isinstance(module.stride, (list, tuple)) else module.stride,
        module.padding[0] if isinstance(module.padding, (list, tuple)) else module.padding,
        module.output_padding[0] if isinstance(module.output_padding, (list, tuple)) else module.output_padding,
        module.dilation[0] if isinstance(module.dilation, (list, tuple)) else module.dilation,
        module.groups,
        'hasBias' if module.bias is not None else '',
    )

    sd = module.state_dict()
    if 'weight' not in sd.keys() and 'weight_g' in sd.keys():
        weight = restore_weighted_norm_inplace(sd['weight_g'], sd['weight_v'])
    else:
        weight = sd['weight']
    bias = sd['bias'] if module.bias is not None else None
    return convert_demonstrate(module, inputs, output)


def convert_Module_BatchNorm2d(module=nn.BatchNorm2d(16, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True), inputs=None, output=None):
    logger.debug('Converting... BatchNorm2d, N {} eps {} momentum {} {} {}',
        module.num_features,
        module.eps,
        module.momentum,
        'affine' if module.affine else '',
        'track_stats' if module.track_running_stats else '',
    )
    sd = module.state_dict()
    scale = sd['weight']
    shift = sd['bias']
    mean = sd['running_mean']
    variance = sd['running_var']
    num_batches_tracked = sd['num_batches_tracked'].item()
    return convert_demonstrate(module, inputs, output)


def convert_Module_BatchNorm1d(module=nn.BatchNorm1d(16, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True), inputs=None, output=None):
    logger.debug('Converting... BatchNorm1d, N {} eps {} momentum {} {} {}',
        module.num_features,
        module.eps,
        module.momentum,
        'affine' if module.affine else '',
        'track_stats' if module.track_running_stats else '',
    )
    sd = module.state_dict()
    scale = sd['weight']
    shift = sd['bias']
    mean = sd['running_mean']
    variance = sd['running_var']
    num_batches_tracked = sd['num_batches_tracked'].item()
    return convert_demonstrate(module, inputs, output)


def convert_Module_GroupNorm(module=nn.GroupNorm(num_groups=1, num_channels=128, eps=1e-5, affine=True), inputs=None, output=None):
    logger.debug('Converting... GroupNorm, G {} C {} eps {} {}',
        module.num_groups,
        module.num_channels,
        module.eps,
        'affine' if module.affine else '',
    )
    sd = module.state_dict()
    scale = sd['weight']
    shift = sd['bias']
    return convert_demonstrate(module, inputs, output)

def convert_Module_InstanceNorm2d(module=nn.InstanceNorm2d(16, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True), inputs=None, output=None):
    logger.debug('Converting... InstanceNorm2d, N {} eps {} momentum {} {} {}',
        module.num_features,
        module.eps,
        module.momentum,
        'affine' if module.affine else '',
        'track_stats' if module.track_running_stats else '',
    )
    sd = module.state_dict()
    scale = sd['weight']
    shift = sd['bias']
    return convert_demonstrate(module, inputs, output)



def convert_torch_nn_functional_max_pool2d(func=F.max_pool2d, inputs=None, output=None, *args, **kwargs):
    def get_torch_nn_functional_max_pool2d_arguments(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        return {'input':input, 'kernel_size':kernel_size, 'stride':kernel_size if stride is None else stride, 'padding':padding, 'ceil_mode':ceil_mode, 'dilation':dilation, 'return_indices':return_indices}
    args_dict = get_torch_nn_functional_max_pool2d_arguments(*args, **kwargs)
    logger.debug('Converting... MaxPool2d, K{}x{} S{}x{} P{}{} D{}x{} ceil?{}',
        args_dict['kernel_size'][0] if isinstance(args_dict['kernel_size'], (list, tuple)) else args_dict['kernel_size'],
        args_dict['kernel_size'][1] if isinstance(args_dict['kernel_size'], (list, tuple)) else args_dict['kernel_size'],
        args_dict['stride'][0] if isinstance(args_dict['stride'], (list, tuple)) else args_dict['stride'],
        args_dict['stride'][1] if isinstance(args_dict['stride'], (list, tuple)) else args_dict['stride'],
        args_dict['padding'][0] if isinstance(args_dict['padding'], (list, tuple)) else args_dict['padding'],
        args_dict['padding'][1] if isinstance(args_dict['padding'], (list, tuple)) else args_dict['padding'],
        args_dict['dilation'][0] if isinstance(args_dict['dilation'], (list, tuple)) else args_dict['dilation'],
        args_dict['dilation'][1] if isinstance(args_dict['dilation'], (list, tuple)) else args_dict['dilation'],
        'Y' if args_dict['ceil_mode'] else 'N',
    )
    return convert_demonstrate(func, inputs, output)


def convert_Module_MaxPool2d(module=nn.MaxPool2d(2, 2), inputs=None, output=None):
    logger.debug('Converting... MaxPool2d, K{}x{} S{}x{} P{}{} D{}x{} ceil?{}',
        module.kernel_size[0] if isinstance(module.kernel_size, (list, tuple)) else module.kernel_size,
        module.kernel_size[1] if isinstance(module.kernel_size, (list, tuple)) else module.kernel_size,
        module.stride[0] if isinstance(module.stride, (list, tuple)) else module.stride,
        module.stride[1] if isinstance(module.stride, (list, tuple)) else module.stride,
        module.padding[0] if isinstance(module.padding, (list, tuple)) else module.padding,
        module.padding[1] if isinstance(module.padding, (list, tuple)) else module.padding,
        module.dilation[0] if isinstance(module.dilation, (list, tuple)) else module.dilation,
        module.dilation[1] if isinstance(module.dilation, (list, tuple)) else module.dilation,
        'Y' if module.ceil_mode else 'N',
    )
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_avg_pool2d(func=F.avg_pool2d, inputs=None, output=None, *args, **kwargs):
    def get_torch_nn_functional_avg_pool2d_arguments(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        return { 'input':input, 'kernel_size':kernel_size, 'stride':kernel_size if stride is None else stride, 'padding':padding, 'ceil_mode':ceil_mode, 'count_include_pad':count_include_pad }
    args_dict = get_torch_nn_functional_avg_pool2d_arguments(*args, **kwargs)
    logger.debug('Converting... AvgPool2d, K{}x{} S{}x{} P{}{} D{}x{} ceil?{} {}',
        args_dict['kernel_size'][0] if isinstance(args_dict['kernel_size'], (list, tuple)) else args_dict['kernel_size'],
        args_dict['kernel_size'][1] if isinstance(args_dict['kernel_size'], (list, tuple)) else args_dict['kernel_size'],
        args_dict['stride'][0] if isinstance(args_dict['stride'], (list, tuple)) else args_dict['stride'],
        args_dict['stride'][1] if isinstance(args_dict['stride'], (list, tuple)) else args_dict['stride'],
        args_dict['padding'][0] if isinstance(args_dict['padding'], (list, tuple)) else args_dict['padding'],
        args_dict['padding'][1] if isinstance(args_dict['padding'], (list, tuple)) else args_dict['padding'],
        args_dict['dilation'][0] if isinstance(args_dict['dilation'], (list, tuple)) else args_dict['dilation'],
        args_dict['dilation'][1] if isinstance(args_dict['dilation'], (list, tuple)) else args_dict['dilation'],
        'Y' if args_dict['ceil_mode'] else 'N',
        'countIncludePad' if args_dict['count_include_pad'] else '',
    )
    return convert_demonstrate(func, inputs, output)


def convert_Module_AvgPool2d(module=nn.AvgPool2d((3, 3), 1, 1), inputs=None, output=None):
    logger.debug('Converting... AvgPool2d, K{}x{} S{}x{} P{}{} ceil?{} {}',
        module.kernel_size[0] if isinstance(module.kernel_size, (list, tuple)) else module.kernel_size,
        module.kernel_size[1] if isinstance(module.kernel_size, (list, tuple)) else module.kernel_size,
        module.stride[0] if isinstance(module.stride, (list, tuple)) else module.stride,
        module.stride[1] if isinstance(module.stride, (list, tuple)) else module.stride,
        module.padding[0] if isinstance(module.padding, (list, tuple)) else module.padding,
        module.padding[1] if isinstance(module.padding, (list, tuple)) else module.padding,
        'Y' if module.ceil_mode else 'N',
        'countIncludePad' if module.count_include_pad else '',
    )
    return convert_demonstrate(module, inputs, output)


def convert_Module_AdaptiveMaxPool2d(module=nn.AdaptiveMaxPool2d((1, 1)), inputs=None, output=None):
    if module.output_size[0] == 1 and module.output_size[1] == 1:
        logger.debug('Converting... MaxPool2d (from Adaptive 1x1), K{}x{} S1x1 P00 D1x1 ceil?N',
            inputs[0].size(2),
            inputs[0].size(3),
        )
        return convert_demonstrate(module, inputs, output)

    logger.debug('Converting... AdaptiveMaxPool2d, output size {}',
        module.output_size,
    )
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_adaptive_avg_pool2d(func=F.adaptive_avg_pool2d, inputs=None, output=None, *args, **kwargs):
    def get_torch_nn_functional_adaptive_avg_pool2d_arguments(input, output_size):
        return { 'input':input, 'output_size':output_size }
    args_dict = get_torch_nn_functional_adaptive_avg_pool2d_arguments(*args, **kwargs)
    output_size = args_dict['output_size'] if isinstance(args_dict['output_size'], (list, tuple)) else (args_dict['output_size'], args_dict['output_size'])
    if output_size[0] == 1 and output_size[1] == 1:
        logger.debug('Converting... AvgPool2d (from Adaptive 1x1), K{}x{} S1x1 P00 D1x1 ceil?N countIncludePad',
            inputs[0].size(2),
            inputs[0].size(3),
        )
        return convert_demonstrate(func, inputs, output)

    logger.debug('Converting... AdaptiveAvgPool2d, output size {}',
        module.output_size if isinstance(module.output_size, (list, tuple)) else (module.output_size, module.output_size),
    )
    return convert_demonstrate(func, inputs, output)


def convert_Module_AdaptiveAvgPool2d(module=nn.AdaptiveAvgPool2d((1, 1)), inputs=None, output=None):
    output_size = module.output_size if isinstance(module.output_size, (list, tuple)) else (module.output_size, module.output_size)
    if output_size[0] == 1 and output_size[1] == 1:
        logger.debug('Converting... AvgPool2d (from Adaptive 1x1), K{}x{} S1x1 P00 D1x1 ceil?N countIncludePad',
            inputs[0].size(2),
            inputs[0].size(3),
        )
        return convert_demonstrate(module, inputs, output)

    logger.debug('Converting... AdaptiveAvgPool2d, output size {}',
        module.output_size if isinstance(module.output_size, (list, tuple)) else (module.output_size, module.output_size),
    )
    return convert_demonstrate(module, inputs, output)


def convert_Module_Linear(module=nn.Linear(4, 8, True), inputs=None, output=None):
    logger.debug('Converting... FullConnection, C{} N{}',
        module.in_features,
        module.out_features,
    )
    sd = module.state_dict()
    weight = sd['weight']
    bias = sd['bias']
    return convert_demonstrate(module, inputs, output)


def convert_Module_FCNorm(module=None, inputs=None, output=None):
    logger.debug('Converting... FullConnectionNorm, C{} N{}',
        module.linear.in_features,
        module.linear.out_features,
    )
    sd = module.state_dict()
    weight = sd['weight']
    bias = sd['bias']
    return convert_demonstrate(module, inputs, output)


def convert_Module_SoftArgMax(module=None, inputs=None, output=None):
    logger.debug('Converting... SoftArgmax, beta {}',
        module.beta,
    )
    return convert_demonstrate(module, inputs, output)


def convert_Module_SoftArgMax1D(module=None, inputs=None, output=None):
    logger.debug('Converting... SoftArgmax1D, beta {}',
        module.beta,
    )
    return convert_demonstrate(module, inputs, output)


def convert_torch_flip(func=torch.flip, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_flip_arguments(input=[], dims=0):
        return { 'input':input, 'dims':dims}
    # ---
    args_dict = get_torch_flip_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... Flip, dims {}',
        args_dict['dims'],
    )
    return convert_demonstrate(func, inputs, output)


def convert_Module_Flip(module=None, inputs=None, output=None):
    logger.debug('Converting... Flip, dims {}',
        module.dims,
    )
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_fold(func=F.fold, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_nn_functional_fold_arguments(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
        return { 'input': input, 'output_size': output_size, 'kernel_size': kernel_size,
        'dilation': dilation, 'padding': padding, 'stride': stride }
    # ---
    args_dict = get_torch_nn_functional_fold_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... Fold, N{} K{} S{} P{} D{}',
        args_dict['output_size'],
        args_dict['kernel_size'],
        args_dict['stride'],
        args_dict['padding'],
        args_dict['dilation'],
    )
    return convert_demonstrate(func, inputs, output)


def convert_Module_LayerNorm(module=nn.LayerNorm(normalized_shape=0, eps=1e-5, elementwise_affine=True), inputs=None, output=None):
    logger.debug('Converting... LayerNorm, shape {} eps {} {}',
        module.normalized_shape,
        module.eps,
        'elemAffine' if module.elementwise_affine else '',
    )
    return convert_demonstrate(module, inputs, output)
