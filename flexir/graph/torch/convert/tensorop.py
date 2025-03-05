
from .common import *

def convert_torch_clamp(func=torch.clamp, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_clamp_arguments(input, min, max, out=None):
        return { 'input':input,
        'min': min if min is not None else float('-Infinity'),
        'max': max if max is not None else float('Infinity')}
    # ---
    args_dict = get_torch_clamp_arguments(*args, **kwargs)
    # ---
    ASSERT(isinstance(args_dict['min'], (float, int)))
    ASSERT(isinstance(args_dict['max'], (float, int)))
    # ---
    return convert_demonstrate(func, inputs, output)

def convert_torch_tensor_clamp(func=torch.Tensor.clamp, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_clamp_arguments(self_input, min, max, out=None):
        return { 'input':self_input, 'min':min, 'max':max }
    # ---
    args_dict = get_torch_tensor_clamp_arguments(*args, **kwargs)
    # ---
    ASSERT(isinstance(args_dict['min'], (float, int)))
    ASSERT(isinstance(args_dict['max'], (float, int)))
    # ---
    return convert_demonstrate(func, inputs, output)



def convert_torch_tensor_expand(func=torch.Tensor.expand, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_expand_arguments(self_input, *shape):
        return { 'input':self_input, 'shape':recurrent_split_arguments(shape) }
    # ---
    args_dict = get_torch_tensor_expand_arguments(*args, **kwargs)
    # ---
    return convert_demonstrate(func, inputs, output)

def convert_torch_tensor_repeat(func=torch.Tensor.repeat, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_repeat_arguments(self_input, *sizes):
        return { 'input':self_input, 'sizes':sizes }
    # ---
    args_dict = get_torch_tensor_repeat_arguments(*args, **kwargs)
    # ---
    return convert_demonstrate(func, inputs, output)

def convert_torch_tensor_expand_as(func=torch.Tensor.expand_as, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_expand_arguments(self_input, other):
        return { 'input':self_input, 'other':other}
    # ---
    args_dict = get_torch_tensor_expand_arguments(*args, **kwargs)
    # ---
    return convert_demonstrate(func, inputs, output)

def convert_torch_bmm(func=torch.bmm, inputs=None, output=None, *args, **kwargs):
    get_torch_bmm_arguments = lambda input, mat2 : {
        'input':input,
        'mat2':mat2
    }
    args_dict = get_torch_bmm_arguments(*args, **kwargs)

    # https://pytorch.org/docs/stable/generated/torch.bmm.html#torch.bmm
    assert(args_dict['input'].dim() == 3)
    assert(args_dict['mat2'].dim() == 3)
    assert(args_dict['input'].shape[0] == args_dict['mat2'].shape[0])

    return convert_demonstrate(func, inputs, output)

def convert_torch_matmul(func=torch.matmul, inputs=None, output=None, *args, **kwargs):
    '''
    https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul
    '''
    get_torch_matmul_arguments = lambda input, other : {
        'input':input,
        'other':other
    }
    args_dict = get_torch_matmul_arguments(*args, **kwargs)
    mat1 = args_dict['input']
    mat2 = args_dict['other']
    mat1_dim = mat1.dim()
    mat2_dim = mat2.dim()

    # do not support broadcasting

    is_1D = lambda dim : (dim == 1)
    if is_1D(mat1_dim) or is_1D(mat2_dim):
        raise NotImplementedError('Failed tried converting matmul to bmm')

    if mat1_dim == mat2_dim:
        # only support: [<same>, M, N], [<same>, N, K]
        batch_dims = mat1_dim - 2
        batch_shapes_mat1 = mat1.shape[: batch_dims]
        batch_shapes_mat2 = mat2.shape[: batch_dims]
        are_same_batches = True
        for i in range(batch_dims):
            if batch_shapes_mat1[i] != batch_shapes_mat2[i]:
                are_same_batches = False
                break
        if not are_same_batches:
            raise NotImplementedError('Failed tried converting matmul to bmm')

    else:   # mat1_dim != mat2_dim
        # only support: [1, ..., 1, M, N], [1, ..., 1, N, K]
        batch_shapes_mat1 = mat1.shape[: (len(mat1.shape) - 2)]
        batch_shapes_mat2 = mat2.shape[: (len(mat2.shape) - 2)]
        is_trivial_batch = True
        for i in range(len(mat1.shape) - 2):
            if batch_shapes_mat1[i] != 1:
                is_trivial_batch = False
                break
        for i in range(len(mat2.shape) - 2):
            if batch_shapes_mat2[i] != 1:
                is_trivial_batch = False
                break
        if not is_trivial_batch:
            raise NotImplementedError('Failed tried converting matmul to bmm')

    return convert_demonstrate(func, inputs, output)

def convert_torch_tensor_add(func=torch.Tensor.add, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_add_arguments(self_inp, other):
        return { 'input':self_inp, 'other':other }
    # ---
    args_dict = get_torch_tensor_add_arguments(*args, **kwargs)
    # ---
    if isinstance(args_dict['other'], (float, int)):
        return convert_demonstrate(func, inputs, output)

    if args_dict['input'].size() == args_dict['other'].size():
        # info = ElementWiseOperatorInfo()
        # info._op_type = ENUM_TYPE_ELEMENTWISE_OP['Add']
        pass
    else:
        # info = BroadcastOperatorInfo()
        # info._op_type = ENUM_TYPE_BROADCAST_OP['Add']
        pass

    return convert_demonstrate(func, inputs, output)

def convert_torch_tensor_operator_iadd(func=torch.Tensor.__iadd__, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_operator_iadd_arguments(self_inp, other):
        return { 'input':self_inp, 'other':other }
    # ---
    args_dict = get_torch_tensor_operator_iadd_arguments(*args, **kwargs)
    # ---
    if isinstance(args_dict['other'], (float, int)):
        return convert_demonstrate(func, inputs, output)

    if args_dict['input'].size() == args_dict['other'].size():
        # info = ElementWiseOperatorInfo()
        # info._op_type = ENUM_TYPE_ELEMENTWISE_OP['Add']
        pass
    else:
        # info = BroadcastOperatorInfo()
        # info._op_type = ENUM_TYPE_BROADCAST_OP['Add']
        pass

    return convert_demonstrate(func, inputs, output)

def convert_torch_tensor_operator_add(func=torch.Tensor.__add__, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_operator_add_arguments(self_inp, other):
        return { 'input':self_inp, 'other':other }
    # ---
    args_dict = get_torch_tensor_operator_add_arguments(*args, **kwargs)
    # ---
    if isinstance(args_dict['other'], (float, int)):
        # info = ActivationInfo()
        # info._activation_type = ENUM_TYPE_ACTIVATION['Linear']
        # info._activation_alpha = float(1.0)
        # info._activation_beta = float(args_dict['other'])
        # info._input_ids = [ str(id(args_dict['input']))]
        # return info
        return convert_demonstrate(func, inputs, output)

    if args_dict['input'].size() == args_dict['other'].size():
        # info = ElementWiseOperatorInfo()
        # info._op_type = ENUM_TYPE_ELEMENTWISE_OP['Add']
        pass
    else:
        # info = BroadcastOperatorInfo()
        # info._op_type = ENUM_TYPE_BROADCAST_OP['Add']
        pass

    return convert_demonstrate(func, inputs, output)

def convert_torch_add(func=torch.add, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_add_arguments(input, other, out=None):
        return { 'input': input, 'other':other, 'out':out }
    # ---
    args_dict = get_torch_add_arguments(*args, **kwargs)
    # ---
    if isinstance(args_dict['other'], (float, int)):
        return convert_demonstrate(func, inputs, output)

    if args_dict['input'].size() == args_dict['other'].size():
        # info = ElementWiseOperatorInfo()
        # info._op_type = ENUM_TYPE_ELEMENTWISE_OP['Add']
        pass
    else:
        # info = BroadcastOperatorInfo()
        # info._op_type = ENUM_TYPE_BROADCAST_OP['Add']
        pass

    return convert_demonstrate(func, inputs, output)


def convert_Module_Add(module=None, inputs=None, output=None):
    logger.debug('Converting... Add')
    return convert_demonstrate(module, inputs, output)

def convert_torch_tensor_operator_sub(func=torch.Tensor.__sub__, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_operator_sub_arguments(self_inp, other):
        return { 'input':self_inp, 'other':other }
    # ---
    args_dict = get_torch_tensor_operator_sub_arguments(*args, **kwargs)
    # ---
    if isinstance(args_dict['other'], (float, int)):
        return convert_demonstrate(func, inputs, output)

    if args_dict['input'].size() == args_dict['other'].size():
        # info = ElementWiseOperatorInfo()
        # info._op_type = ENUM_TYPE_ELEMENTWISE_OP['Sub']
        pass
    else:
        # info = BroadcastOperatorInfo()
        # info._op_type = ENUM_TYPE_BROADCAST_OP['Sub']
        pass

    return convert_demonstrate(func, inputs, output)

def convert_torch_mul(func=torch.mul, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_mul_arguments(input, other, out=None):
        return { 'input': input, 'other':other, 'out':out }
    # ---
    args_dict = get_torch_mul_arguments(*args, **kwargs)
    # ---
    if isinstance(args_dict['other'], (float, int)):
        return convert_demonstrate(func, inputs, output)

    if args_dict['input'].size() == args_dict['other'].size():
        # info = ElementWiseOperatorInfo()
        # info._op_type = ENUM_TYPE_ELEMENTWISE_OP['Mul']
        pass
    else:
        # info = BroadcastOperatorInfo()
        # info._op_type = ENUM_TYPE_BROADCAST_OP['Mul']
        pass

    return convert_demonstrate(func, inputs, output)

def convert_torch_tensor_operator_mul(func=torch.Tensor.__mul__, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_operator_mul_arguments(self_inp, other):
        return { 'input':self_inp, 'other':other }
    # ---
    args_dict = get_torch_tensor_operator_mul_arguments(*args, **kwargs)
    # ---
    if isinstance(args_dict['other'], (float, int)):
        return convert_demonstrate(func, inputs, output)

    if args_dict['input'].size() == args_dict['other'].size():
        # info = ElementWiseOperatorInfo()
        # info._op_type = ENUM_TYPE_ELEMENTWISE_OP['Mul']
        pass
    else:
        # info = BroadcastOperatorInfo()
        # info._op_type = ENUM_TYPE_BROADCAST_OP['Mul']
        pass

    return convert_demonstrate(func, inputs, output)

def convert_torch_tensor_operator_div(func=torch.Tensor.__truediv__, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_operator_div_arguments(self_inp, other):
        return { 'input':self_inp, 'other':other }
    # ---
    args_dict = get_torch_tensor_operator_div_arguments(*args, **kwargs)
    # ---
    if isinstance(args_dict['other'], (float, int)):
        return convert_demonstrate(func, inputs, output)

    if args_dict['input'].size() == args_dict['other'].size():
        # info = ElementWiseOperatorInfo()
        # info._op_type = ENUM_TYPE_ELEMENTWISE_OP['Div']
        pass
    else:
        # info = BroadcastOperatorInfo()
        # info._op_type = ENUM_TYPE_BROADCAST_OP['Div']
        pass

    return convert_demonstrate(func, inputs, output)

def convert_torch_tensor_operator_pow(func=torch.Tensor.__pow__, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_operator_pow_arguments(self_inp, other):
        return { 'input':self_inp, 'other':other }
    # ---
    args_dict = get_torch_tensor_operator_pow_arguments(*args, **kwargs)
    # ---
    if not isinstance(args_dict['other'], (float, int)):
        raise NotImplementedError

    return convert_demonstrate(func, inputs, output)

def convert_torch_tensor_operator_neg(func=torch.Tensor.__neg__, inputs=None, output=None, *args, **kwargs):
    # ---
    def get_torch_tensor_operator_neg_arguments(self_inp):
        return { 'input':self_inp}
    # ---
    args_dict = get_torch_tensor_operator_neg_arguments(*args, **kwargs)
    return convert_demonstrate(func, inputs, output)


def convert_torch_tensor_operator_getitem(kernel=torch.Tensor.__getitem__, inputs=[], outputs=[], *args, **kwargs):
    # ---
    def get_torch_tensor_operator_getitem_arguments(self_inp=torch.ones(2,3,4,5,6,7), key=0):
        if isinstance(key, (int, float)):
            return { 'input':self_inp, 'py_sampling_infos': [key] }

        return { 'input':self_inp, 'py_sampling_infos': key }
    # ---
    args_dict = get_torch_tensor_operator_getitem_arguments(*args, **kwargs)
    # ---
    return convert_demonstrate(kernel, inputs, output)

def convert_torch_tensor_operator_setitem(kernel=torch.Tensor.__setitem__, inputs=[], outputs=[], *args, **kwargs):
    # ---
    def get_torch_tensor_operator_setitem_arguments(self_inp=torch.ones(2,3,4,5,6,7), key=0, tensor_to_set=torch.zeros(5,6,7)):
        if isinstance(key, (int, float)):
            return { 'input':self_inp, 'py_sampling_infos': [key], 'tensor_to_set':tensor_to_set }

        return { 'input':self_inp, 'py_sampling_infos': key, 'tensor_to_set':tensor_to_set }
    # ---
    return convert_demonstrate(kernel, inputs, output)


def convert_torch_dimension_sampling_info(input_shape=[], py_sampling_infos=[]):
    dims = len(input_shape)
    dim_offset = 0
    # sampling_infos = { }
    num_ellipsis = 0
    for k in range(len(py_sampling_infos)):
        py_sampling_info = py_sampling_infos[k]
        if isinstance(py_sampling_info, slice):
            sfrom = py_sampling_info.start
            sto = py_sampling_info.stop
            sstep = py_sampling_info.step
            if sfrom is None:
                sfrom = 0
            else:
                if sfrom < 0:
                    sfrom += input_shape[dim_offset]


            if sto is None:
                sto = -1
            else:
                if sto < 0 and sto != -1:
                    sto += input_shape[dim_offset]


            if sstep is None:
                sstep = 1
            else:
                if sstep < 0:
                    sstep = 1


            # sampling_infos[str(k)] = TensorDimemsionSamplingInfo(sampling_type = ENUM_TYPE_GETITEM_DIM_SAMPLING['Slice'], data=[sfrom, sto, sstep])
            dim_offset += 1
        elif isinstance(py_sampling_info, Ellipsis.__class__):
            # sampling_infos[str(k)] = TensorDimemsionSamplingInfo(sampling_type = ENUM_TYPE_GETITEM_DIM_SAMPLING['Ellipsis'], data=[])
            dim_offset = dims - (len(py_sampling_infos) - 1 - k)
            num_ellipsis += 1
        elif isinstance(py_sampling_info, int):
            # sampling_infos[str(k)] = TensorDimemsionSamplingInfo(sampling_type = ENUM_TYPE_GETITEM_DIM_SAMPLING['Int'], data=[py_sampling_info])
            dim_offset += 1
        elif isinstance(py_sampling_info, (list, tuple)):
            # sampling_infos[str(k)] = TensorDimemsionSamplingInfo(sampling_type = ENUM_TYPE_GETITEM_DIM_SAMPLING['Array'], data=py_sampling_info)
            dim_offset += 1
        else:
            raise RuntimeError('Unsupport Type of Dimension Sampling.')

    ASSERT(num_ellipsis <= 1)
    # return sampling_infos
    return convert_demonstrate(func, inputs, output)
    #TODO
