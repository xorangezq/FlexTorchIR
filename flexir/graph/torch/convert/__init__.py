# -*- coding: utf-8 -*-

from .common import *
from .activation import *
from .shape import *
from .tensorop import *
from .rnn import *
from .llm import *


AVALIABLE_FUNCTION_CONVERTER = {
    # torch ...
    torch.cat: convert_torch_cat,
    torch.add: convert_torch_add,
    torch.mul: convert_torch_mul,
    torch.sigmoid: convert_torch_sigmoid,
    torch.reshape: convert_torch_reshape,
    torch.transpose: convert_torch_transpose,
    torch.unsqueeze: convert_torch_unsqueeze,
    torch.squeeze: convert_torch_squeeze,
    torch.clamp: convert_torch_clamp,
    torch.bmm: convert_torch_bmm,
    torch.matmul: convert_torch_matmul,
    torch.tanh: convert_torch_tanh,
    torch.flip: convert_torch_flip,
    torch.stack: convert_torch_stack,
    # torch.nn.functional ...
    F.conv2d: convert_torch_nn_functional_conv2d,
    F.max_pool2d: convert_torch_nn_functional_max_pool2d,
    F.avg_pool2d: convert_torch_nn_functional_avg_pool2d,
    F.adaptive_avg_pool2d: convert_torch_nn_functional_adaptive_avg_pool2d,
    F.upsample: convert_torch_nn_functional_upsample,
    F.interpolate: convert_torch_nn_functional_interpolate,
    F.pixel_shuffle: convert_torch_nn_functional_pixel_shuffle,
    F.relu: convert_torch_nn_functional_relu,
    F.relu_: convert_torch_nn_functional_relu,
    F.relu6: convert_torch_nn_functional_relu6,
    F.leaky_relu: convert_torch_nn_functional_leaky_relu,
    # F.leaky_relu_: convert_torch_nn_functional_leaky_relu, # Warning! Unable to convert inplace op
    F.tanh: convert_torch_nn_functional_tanh,
    F.sigmoid: convert_torch_nn_functional_sigmoid,
    F.softmax: convert_torch_nn_functional_softmax,
    F.hardswish: convert_torch_nn_functional_hswish,
    F.hardsigmoid: convert_torch_nn_functional_hsigmoid,
    F.prelu: convert_torch_nn_functional_prelu,
    F.elu: convert_torch_nn_functional_elu,
    F.silu: convert_torch_nn_functional_silu,
    F.pad: convert_torch_nn_functional_pad,
    F.grid_sample: convert_torch_nn_functional_grid_sample,
    F.affine_grid: convert_torch_nn_functional_affine_grid,
    F.fold: convert_torch_nn_functional_fold,
    # torch.Tensor ...
    torch.Tensor.transpose: convert_torch_tensor_transpose,
    torch.Tensor.view: convert_torch_tensor_view,
    torch.Tensor.reshape: convert_torch_tensor_reshape,
    torch.Tensor.permute: convert_torch_tensor_permute,
    torch.Tensor.squeeze: convert_torch_tensor_squeeze,
    torch.Tensor.unsqueeze: convert_torch_tensor_unsqueeze,
    torch.Tensor.__getitem__: convert_torch_tensor_operator_getitem,
    torch.Tensor.__setitem__: convert_torch_tensor_operator_setitem,
    torch.Tensor.add: convert_torch_tensor_add,
    torch.Tensor.__add__: convert_torch_tensor_operator_add,
    torch.Tensor.__sub__: convert_torch_tensor_operator_sub,
    torch.Tensor.__iadd__: convert_torch_tensor_operator_iadd,
    torch.Tensor.__mul__: convert_torch_tensor_operator_mul,
    torch.Tensor.__rmul__: convert_torch_tensor_operator_mul,
    torch.Tensor.__pow__: convert_torch_tensor_operator_pow,
    torch.Tensor.__truediv__: convert_torch_tensor_operator_div,
    torch.Tensor.__neg__: convert_torch_tensor_operator_neg,
    torch.Tensor.clamp: convert_torch_tensor_clamp,
    torch.Tensor.expand: convert_torch_tensor_expand,
    torch.Tensor.expand_as: convert_torch_tensor_expand_as,
    torch.Tensor.bmm: convert_torch_bmm,
    torch.Tensor.matmul: convert_torch_matmul,
    torch.Tensor.repeat: convert_torch_tensor_repeat
}

AVALIABLE_MODULE_CONVERTER = {
    # pytorch origin modules:
    torch.nn.modules.conv.Conv2d.__name__: convert_Module_Conv2d,
    torch.nn.modules.conv.Conv1d.__name__: convert_Module_Conv1d,
    torch.nn.modules.conv.ConvTranspose2d.__name__: convert_Module_ConvTranspose2d,
    torch.nn.modules.conv.ConvTranspose1d.__name__: convert_Module_ConvTranspose1d,
    torch.nn.modules.batchnorm.BatchNorm2d.__name__: convert_Module_BatchNorm2d,
    torch.nn.modules.batchnorm.BatchNorm1d.__name__: convert_Module_BatchNorm1d,
    torch.nn.modules.GroupNorm.__name__: convert_Module_GroupNorm,
    torch.nn.modules.instancenorm.InstanceNorm2d.__name__: convert_Module_InstanceNorm2d,
    torch.nn.modules.activation.ReLU.__name__: convert_Module_ReLU,
    torch.nn.modules.activation.ReLU6.__name__: convert_Module_ReLU6,
    torch.nn.modules.activation.LeakyReLU.__name__: convert_Module_LeakyReLU,
    torch.nn.modules.pooling.MaxPool2d.__name__: convert_Module_MaxPool2d,
    torch.nn.modules.pooling.AvgPool2d.__name__: convert_Module_AvgPool2d,
    torch.nn.modules.upsampling.Upsample.__name__: convert_Module_Upsample,
    torch.nn.modules.upsampling.UpsamplingNearest2d.__name__: convert_Module_UpsamplingNearest2d,
    torch.nn.modules.activation.Tanh.__name__: convert_Module_Tanh,
    torch.nn.modules.activation.Sigmoid.__name__: convert_Module_Sigmoid,
    torch.nn.modules.Linear.__name__: convert_Module_Linear,
    torch.nn.modules.AdaptiveAvgPool2d.__name__ : convert_Module_AdaptiveAvgPool2d,
    torch.nn.modules.AdaptiveMaxPool2d.__name__ : convert_Module_AdaptiveMaxPool2d,
    torch.nn.modules.Identity.__name__: convert_Module_Identity,
    torch.nn.Softmax.__name__: convert_Module_Softmax,
    torch.nn.modules.RNNCell.__name__: convert_Module_RNNCell,
    torch.nn.modules.RNN.__name__: convert_Module_RNN,
    torch.nn.modules.LSTMCell.__name__: convert_Module_LSTMCell,
    torch.nn.modules.LSTM.__name__: convert_Module_LSTM,
    torch.nn.modules.GRUCell.__name__: convert_Module_GRUCell,
    torch.nn.modules.GRU.__name__: convert_Module_GRU,
    torch.nn.modules.activation.Hardswish.__name__: convert_Module_HSwish,
    torch.nn.modules.activation.Hardsigmoid.__name__: convert_Module_HSigmoid,
    torch.nn.modules.activation.PReLU.__name__: convert_Module_PReLU,
    torch.nn.modules.activation.ELU.__name__: convert_Module_ELU,
    torch.nn.modules.activation.SiLU.__name__: convert_Module_SiLU,
    torch.nn.modules.PixelShuffle.__name__: convert_Module_PixelShuffle,
    torch.nn.modules.ZeroPad2d.__name__: convert_Module_ZeroPad2d,
    torch.nn.modules.ReflectionPad1d.__name__: convert_Module_ReflectionPad1d,
    torch.nn.LayerNorm.__name__: convert_Module_LayerNorm,
    torch.nn.Embedding.__name__: convert_Module_Embedding,
    torch.nn.modules.MultiheadAttention.__name__: convert_Module_MultiheadAttention,

    # custom modules:
    'SoftArgMax': convert_Module_SoftArgMax,
    'SoftArgMax1D': convert_Module_SoftArgMax1D,
    'FCNorm': convert_Module_FCNorm,
    'MultiHeadAttentionRelativePosition': convert_Module_MultiHeadAttentionRelativePosition,

    # register new modules here ...
}
