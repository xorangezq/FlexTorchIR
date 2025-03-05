
from .common import *

def convert_torch_unsqueeze(func=torch.unsqueeze, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_unsqueeze_arguments(self_inp, dim, out=None):
        return { 'input':self_inp, 'dim': dim }
    # ---
    args_dict = get_torch_unsqueeze_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... unsqueeze, dim {}',
        args_dict['dim']
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_tensor_unsqueeze(func=torch.Tensor.unsqueeze, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_unsqueeze_arguments(self_inp, dim=None):
        return { 'input':self_inp, 'dim': dim }
    # ---
    args_dict = get_torch_tensor_unsqueeze_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... unsqueeze, dim {}',
        args_dict['dim']
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_squeeze(func=torch.squeeze, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_squeeze_arguments(inp, dim=None, out=None):
        return { 'input':inp, 'dim': dim }
    # ---
    args_dict = get_torch_squeeze_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... Squeeze, dim {}',
        args_dict['dim'] if args_dict['dim'] else -1,
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_tensor_squeeze(func=torch.Tensor.squeeze, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_squeeze_arguments(self_inp, dim=None):
        return { 'input':self_inp, 'dim': dim }
    # ---
    args_dict = get_torch_tensor_squeeze_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... Squeeze, dim {}',
        args_dict['dim'] if args_dict['dim'] else -1,
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_transpose(func=torch.Tensor.transpose, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_transpose_arguments(inp, dim0, dim1):
        return { 'input': inp, 'dim0': dim0, 'dim1': dim1 }
    # ---
    args_dict = get_torch_transpose_arguments(*args, **kwargs)
    # ---
    dims = args_dict['input'].dim()
    axes = list(range(dims))
    axes[args_dict['dim1']] = args_dict['dim0']
    axes[args_dict['dim0']] = args_dict['dim1']
    axes = [d if d >= 0 else (d + len(axes)) for d in axes]
    logger.debug('Converting... Transpose, axes {}',
        axes,
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_tensor_transpose(func=torch.Tensor.transpose, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_transpose_arguments(self_inp, dim0, dim1):
        return { 'input': self_inp, 'dim0': dim0, 'dim1': dim1 }
    # ---
    args_dict = get_torch_tensor_transpose_arguments(*args, **kwargs)
    # ---
    dims = args_dict['input'].dim()
    axes = list(range(dims))
    axes[args_dict['dim1']] = args_dict['dim0']
    axes[args_dict['dim0']] = args_dict['dim1']
    axes = [d if d >= 0 else (d + len(axes)) for d in axes]
    logger.debug('Converting... Transpose, axes {}',
        axes,
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_tensor_permute(func=torch.Tensor.permute, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_permute_arguments(self_inp, *dims):
        return { 'input':self_inp, 'dims':recurrent_split_arguments(dims) }
    # ---
    args_dict = get_torch_tensor_permute_arguments(*args, **kwargs)
    # ---
    axes = args_dict['dims']
    axes = [d if d >= 0 else (d + len(axes)) for d in axes]
    logger.debug('Converting... Permute, axes {}',
        axes,
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_reshape(func=torch.reshape, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_reshape_arguments(input, shape):
        return { 'input':input, 'shape':shape }
    # ---
    args_dict = get_torch_reshape_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... Reshape, to {}',
        [s for s in outputs.shape],
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_tensor_reshape(func=torch.Tensor.reshape, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_reshape_arguments(self_input, *shape):
        return { 'input':self_input, 'shape':recurrent_split_arguments(shape) }
    # ---
    args_dict = get_torch_reshape_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... Reshape, to {}',
        [s for s in outputs.shape],
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_tensor_view(func=torch.Tensor.view, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_view_arguments(self_input, *shape):
        return { 'input':self_input, 'shape':recurrent_split_arguments(shape) }
    # ---
    args_dict = get_torch_reshape_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... Reshape(view), to {}',
        [s for s in outputs.shape],
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_cat(func=torch.cat, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_cat_arguments(tensors=[], dim=0, out=None):
        return { 'tensors':tensors, 'dim':dim, 'out':out }
    # ---
    args_dict = get_torch_cat_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... cat, dim {}',
        args_dict['dim'],
    )
    return convert_demonstrate(func, inputs, output)

def convert_torch_stack(func=torch.stack, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_stack_arguments(tensors=[], dim=0):
        return { 'tensors':tensors, 'dim':dim }
    # ---
    args_dict = get_torch_stack_arguments(*args, **kwargs)
    # ---
    logger.debug('Converting... stack, dim {}',
        args_dict['dim'],
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_nn_functional_pixel_shuffle(func=F.pixel_shuffle, inputs=None, output=None, *args, **kwargs):
    def get_torch_nn_functional_pixel_shuffle_arguments(input, upscale_factor):
        return { 'input':input, 'upscale_factor': upscale_factor }
    args_dict = get_torch_nn_functional_pixel_shuffle_arguments(*args, **kwargs)
    logger.debug('Converting... PixelShuffle, upscaling {}x',
        args_dict['upscale_factor'],
    )
    return convert_demonstrate(func, inputs, output)


def convert_Module_PixelShuffle(module=nn.PixelShuffle, inputs=None, output=None):
    logger.debug('Converting... PixelShuffle, upscaling {}x',
        module.upscale_factor,
    )
    return convert_demonstrate(module, inputs, output)


# NN_UPSAMPLE_MODE_MAP_TO_ENUM_TYPE_UPSAMPLE[args_dict['mode']]
# NN_UPSAMPLE_MODE_MAP_TO_ENUM_TYPE_UPSAMPLE = {
#     'nearest': ENUM_TYPE_UPSAMPLE['Nearest'],
#     'linear': ENUM_TYPE_UPSAMPLE['Bilinear'],
#     'bilinear': ENUM_TYPE_UPSAMPLE['Bilinear'],
#     'bicubic': ENUM_TYPE_UPSAMPLE['Bicubic'],
# }
def convert_torch_nn_functional_upsample(func=F.upsample, inputs=None, output=None, *args, **kwargs): # before v1.0.0
    def getUpsampleFunctionArguments(input, size=None, scale_factor=None, mode='nesrest', align_corners=False):
        return { 'input':input, 'size': size, 'scale_factor': scale_factor, 'mode': mode, 'align_corners': align_corners }
    args_dict = getUpsampleFunctionArguments(*args, **kwargs)
    logger.debug('Converting... Upsample, type \'{}\' scale{}x{} align?{}',
        args_dict['mode'],
        float(args_dict['scale_factor'] if args_dict['scale_factor'] is not None else args_dict['size'][1] / args_dict['input'].size()[3]),
        float(args_dict['scale_factor'] if args_dict['scale_factor'] is not None else args_dict['size'][0] / args_dict['input'].size()[2]),
        'Y' if args_dict['align_corners'] else 'N',
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_nn_functional_interpolate(func, inputs, output, *args, **kwargs):
    def getInterpolateFunctionArguments(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        return { 'input':input, 'size': size, 'scale_factor': scale_factor, 'mode': mode, 'align_corners': align_corners, 'recompute_scale_factor': recompute_scale_factor }
    args_dict = getInterpolateFunctionArguments(*args, **kwargs)
    logger.debug('Converting... Upsample(Interpolate), type \'{}\' scale{}x{} align?{}',
        args_dict['mode'],
        float(args_dict['scale_factor'] if args_dict['scale_factor'] is not None else args_dict['size'][1] / args_dict['input'].size()[3]),
        float(args_dict['scale_factor'] if args_dict['scale_factor'] is not None else args_dict['size'][0] / args_dict['input'].size()[2]),
        'Y' if args_dict['align_corners'] else 'N',
    )
    return convert_demonstrate(func, inputs, output)


def convert_Module_UpsamplingNearest2d(module=nn.UpsamplingNearest2d(scale_factor=2), inputs=None, output=None):
    logger.debug('Converting... Upsample(Nearest2d), type \'nearest\' scale{}x{} align?N',
        float(module.scale_factor),
        float(module.scale_factor),
    )
    return convert_demonstrate(module, inputs, output)


def convert_Module_Upsample(module=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=2), inputs=None, output=None):
    logger.debug('Converting... Upsample, type \'{}\' scale{}x{} align?{}',
        module.mode,
        float(module.scale_factor[0]) if isinstance(module.scale_factor, tuple) else float(module.scale_factor),
        float(module.scale_factor[1]) if isinstance(module.scale_factor, tuple) else float(module.scale_factor),
        'Y' if module.align_corners else 'N',
    )
    return convert_demonstrate(module, inputs, output)


def convert_torch_nn_functional_affine_grid(func=F.affine_grid, inputs=None, output=None, *args, **kwargs):
    def get_f_affine_grid_arguments(theta, size, align_corners=None):
        return {
            'theta'         : theta,    # is inputs[0]
            'size'          : size,
            'align_corners' : align_corners,
        }
    args_dict = get_f_affine_grid_arguments(*args, **kwargs)
    logger.debug('Converting... AffineGrid, mode {} padding {} align?{}',
        args_dict['mode'],
        args_dict['padding_mode'],
        'Y' if args_dict['align_corners'] else 'N',
    )
    return convert_demonstrate(func, inputs, output)


def convert_torch_nn_functional_grid_sample(func=F.grid_sample, inputs=None, output=None, *args, **kwargs):
    def get_f_grid_sample_arguments(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
        return {
            'input'         : input,    # is inputs[0]
            'grid'          : grid,     # is inputs[1]
            'mode'          : mode,
            'padding_mode'  : padding_mode,
            'align_corners' : align_corners,
        }
    args_dict = get_f_grid_sample_arguments(*args, **kwargs)
    logger.debug('Converting... GridSample, mode {} padding {} align?{}',
        args_dict['mode'],
        args_dict['padding_mode'],
        'Y' if args_dict['align_corners'] else 'N',
    )
    return convert_demonstrate(func, inputs, output)
