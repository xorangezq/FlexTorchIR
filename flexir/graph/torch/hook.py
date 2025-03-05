#!/usr/bin/env python
# coding: utf-8

from contextlib import contextmanager

import torch
from torch import Tensor
from torch.autograd import Variable

from flexir.utilities.logger import logger, ASSERT

class ConvertContext:

    def __init__(self):
        self.fnhook_depth = 0

        self.module_prehooks = []
        self.module_posthooks = []

        self.dict_origin_F = {}
        self.FUNCTION_OPS_POOL = [
            'torch.nn.functional',
            'torch.Tensor',
            'torch',
            'torch._C._nn'
        ]

        # heuristic, skip anything that is not an op, or we're not interested in
        self.FUNCTION_OPS_SKIP = [
            'torch.get_default_dtype',
            'torch.Tensor.clone',
            'torch.Tensor.detach',
            'torch.Tensor.numpy',
            'torch.Tensor.item',
            'torch.Tensor.__getattribute__',
            'torch.Tensor.numel',
            'torch.Tensor.unbind',
            'torch.Tensor.__iter__',
            'torch.Tensor.tolist',
            'torch.Tensor.__deepcopy__',

            'torch.nn.functional.hardtanh', # torch.nn.modules.activation.ReLU6 uses    # Onnx.Clip

            # +section
            # skip torch.xxx which is duplicated in torch.nn.functional.xxx
            'torch.batch_norm',
            'torch.max_pool2d',
            'torch.relu',
            'torch.relu_',  # in-place version
            'torch.grid_sampler',
            # -section

            # +section
            # functions listed here is a valid op, but is already handled by other methods,
            # thus would print warning but the conversion should be successful.
            'torch.nn.functional.batch_norm',   # by torch.nn.modules.batchnorm.BatchNorm2d
            'torch.nn.functional.linear',       # by module FC ?
            'torch.addmm',                      # maybe we should use this and skip torch.Tensor.__iadd__ etc...
            'torch.affine_grid_generator',      # torch.nn.functional.affine_grid
            # -section

            # converting Tensor.__setitem__ is supported. But it seems verbose so default off. Turn it on when you need it.
            'torch.Tensor.__setitem__',
        ]


    def setup(self, module, torch_ir):
        '''
        TorchIR() is managed by caller
        '''
        self.torch_ir = torch_ir
        self.register_hooks(module)
        self.fnhook_depth = 0
        self.tensor_incarnations = {}

    def teardown(self):
        self.remove_hooks()


    def register_hooks(self, module):
        self.register_module_hooks(module)
        self.register_function_hooks()

    def remove_hooks(self):
        self.remove_module_hooks()
        self.remove_function_hooks()


    def register_module_blackbox(self, mod):
        def forward_pre_hook(module, inputs):
            self.fnhook_depth += 1
        def forward_post_hook(module, inputs, output):
            inputs = list(inputs)
            self.hook_module_forward(module, inputs, output)
            self.fnhook_depth -= 1

        pre_hook = mod.register_forward_pre_hook(forward_pre_hook)
        post_hook = mod.register_forward_hook(forward_post_hook)
        self.module_prehooks.append(pre_hook)
        self.module_posthooks.append(post_hook)

    @contextmanager
    def function_blackbox(self):
        try:
            self.fnhook_depth += 1
            yield
        finally:
            self.fnhook_depth -= 1


    def register_module_hooks(self, module, ignore_module_list=['Sequential']):
        # add hook into modules that is not container
        ignore_sub_modules = []
        for mod in module.modules():
            if mod.__class__.__name__ in ['MultiHeadAttentionRelativePosition']:
                self.register_module_blackbox(mod)
                for sub_mod in mod.modules():
                    ignore_sub_modules.append(sub_mod)

        for mod in module.modules():
            if mod in ignore_sub_modules:
                continue

            if mod.__class__.__name__ in ['FCNorm']:
                self.register_module_blackbox(mod)

            if len(list(mod.modules())) == 1:
                if mod.__class__.__name__ in ignore_module_list:
                    continue

                if mod.__class__.__name__ in [
                    'Dropout2d',
                    'Dropout'
                ]:
                    if mod.inplace == False:
                        mod.inplace = True
                    continue

                if mod.__class__.__name__ in [
                    'ELU',
                    'Hardshrink',
                    'Hardsigmoid',
                    'Hardtanh',
                    'Hardswish',
                    'LeakyReLU',
                    'LogSigmoid',
                    'MultiheadAttention',
                    'PReLU',
                    'ReLU',
                    'ReLU6',
                    'RReLU',
                    'SELU',
                    'CELU',
                    'GELU',
                    'Sigmoid',
                    'SiLU',
                    'Softplus',
                    'Softshrink',
                    'Softsign'
                    'Tanh',
                    'Tanhshrink',
                    'Threshold',
                ]:
                    if hasattr(mod, 'inplace'):
                        if mod.inplace == True:
                            mod.inplace = False

                self.register_module_blackbox(mod)

            elif len(list(mod.modules())) == 2:
                logger.info(mod.__class__.__name__)
                if mod.__class__.__name__ == 'MultiheadAttention':
                    self.register_module_blackbox(mod)


    def remove_module_hooks(self):
        [hook.remove() for hook in self.module_prehooks]
        [hook.remove() for hook in self.module_posthooks]


    def extract_functions_to_hook(self, pymodulename):
        import types

        pymodule = eval(pymodulename)

        extracted_functions = []
        for k in dir(pymodule):
            fullname = '%s.%s' % (pymodulename, k)
            if fullname in self.FUNCTION_OPS_SKIP:
                continue

            is_private_attr = lambda attrname: len(attrname) >= 2 and attrname[0] == '_' and attrname[1] != '_'
            if is_private_attr(k):
                # let's skip xxx._private_attr here
                continue

            field = getattr(pymodule, k)
            if not isinstance(field, (types.BuiltinFunctionType, types.FunctionType, types.MethodDescriptorType, types.WrapperDescriptorType)):
                # logger.info('filter by type:', k, str(type(field)))
                continue

            # logger.info('found builtin or function', fullname)
            extracted_functions.append( fullname )
        return extracted_functions


    def register_function_hooks(self):
        def closure_hook_funcop(funcname):
            def hook_fn(*args, **kwargs):
                nonlocal funcname
                origin_output = self.dict_origin_F[funcname](*args, **kwargs)

                # we mostly want type(origin_output) == torch.Tensor
                # being conservative here and only skip types not interested
                types_not_interested = tuple((
                    bool,
                    int,
                    str,
                    torch.Size
                ))
                if isinstance(origin_output, types_not_interested):
                    return origin_output
                # if not isinstance(origin_output, torch.Tensor):
                #     logger.info('* Unexpected: fn {fnname} produce result type {resulttype}'.format(fnname=funcname, resulttype=type(origin_output)))

                origin_input = args[0]
                if not isinstance(origin_input, torch.Tensor):
                    if isinstance(origin_input, (list, tuple)) and isinstance(origin_input[0], torch.Tensor): # for torch.cat
                        logger.info(funcname)
                    else:
                        return origin_output

                try:
                    output = origin_output
                    # nothing to do with TorchIR.tensorID(), need to know if the same memory
                    if id(output) == id(origin_input):
                        # inplace
                        output = origin_output.clone()

                    if self.fnhook_depth <= 0:
                        with self.function_blackbox():
                            self.hook_function_forward(self.dict_origin_F[funcname], output, *args, **kwargs)
                    # else:
                    #     logger.info('BLOCKED %s by fnhook_depth %d' % (funcname, self.fnhook_depth))

                    return output
                except Exception as e:
                    logger.info('* Surprise when hook function {fnname}: \'{errmsg}\''.format(errmsg=str(e), fnname=funcname))
                    return origin_output
            exec('%s = hook_fn' % funcname)

        for pool in self.FUNCTION_OPS_POOL:
            functions = self.extract_functions_to_hook(pool)
            for fn in functions:
                origin_f = 'self.dict_origin_F[\'%s\']' % fn
                exec('%s = %s' % (origin_f, fn))
                closure_hook_funcop(fn)

    def remove_function_hooks(self):
        def unhook_funcop(funcname):
            if funcname not in self.dict_origin_F.keys():
                return
            origin_f = 'self.dict_origin_F[\'%s\']' % funcname
            exec('%s = %s' % (funcname, origin_f))
            del self.dict_origin_F[funcname]

        for pool in self.FUNCTION_OPS_POOL:
            functions = self.extract_functions_to_hook(pool)
            for fn in functions:
                unhook_funcop(fn)


    '''
    Tensor Incarnation

    Idea:
        mostly from Tensor.__setitem__, e.g.:
        >>> y[:, 0, :, :] = F.elu(y[:, 0, :, :])
        >>> y[:, 1, :, :] = F.elu(y[:, 1, :, :])
        >>> (continue use of y)

    Problem:
        Tensor.__setitem__ does not have output, and changes are made inplace to input tensor
        while fn(inplace=True) has output and id(output) == id(input) (which makes it easy to fool consumers by returning a copy)

        Each call of Tensor.__setitem__ yields a new tensor, but all subsequent module/function has the original tensor for input

    Design:
        use count to break loop into DAG

    Interface:
        Each time a function's id(output) is None, we call tensor_new_incarnation()
        And we check all inputs for modules and functions, replacing any with their latest 'incarnation'

    About execution sequence:
        The best we can do is to take pytorch compute graph's depencency for granted and believe:
        The sequence of executing of our fn hooks is guaranteed to be no-break for data

    Note:
        id() has nothing to do with TorchIR.tensor_ID()
        we're only interested in identification of memory
    '''
    def tensor_new_incarnation(self, tensor):
        self.tensor_incarnations[id(tensor)] = tensor.clone()
        return self.tensor_incarnations[id(tensor)]

    def tensor_find_incarnation(self, tensor):
        if id(tensor) in self.tensor_incarnations:
            return self.tensor_incarnations[id(tensor)]
        else:
            return tensor


    def hook_module_forward(self, module, inputs, outputs):
        output = None
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
        elif isinstance(outputs, (Variable, Tensor)):
            output = outputs
        else:
            ASSERT(False, 'unknow output type: {}'.format(type(outputs)))
        inputs = [self.tensor_find_incarnation(_input) for _input in inputs]
        # logger.info('module forward post hook for module %s with\n\tinput  %s\n\toutput %s' % (
        #     str(module),
        #     ', '.join([str(id(_input)) + '|' + 'x'.join([str(i) for i in _input.shape]) for _input in inputs]),
        #     str(id(output)) + '|' + 'x'.join([str(i) for i in output.shape]))
        # )
        self.torch_ir.convert_module(module, inputs, output)
        return True

    def hook_function_forward(self, func, output, *args, **kwargs):
        torch_inputs =  [a for a in args            if isinstance(a, (Variable, Tensor))]
        torch_inputs += [a for a in kwargs.values() if isinstance(a, (Variable, Tensor))]
        # for <torch.cat> func whose inputs are a sequence of tensor,
        # it should be suited for similar function as well
        for arg in args:
            if isinstance(arg, (list, tuple)):
                torch_inputs += [a for a in arg if isinstance(a, (Variable, Tensor))]
        inputs = [self.tensor_find_incarnation(_input) for _input in torch_inputs]
        if output is None:
            # such as Tensor.__setitem__
            # when a torch function has no output, we think the output is it's input's new incarnation
            output = self.tensor_new_incarnation(torch_inputs[0])
        # logger.info('Func %s\n\tinput  %s\n\toutput %s' % (
        #     func,
        #     ', '.join([str(id(_input)) + '|' + 'x'.join([str(i) for i in _input.shape]) for _input in inputs]),
        #     str(id(output)) + '|' + 'x'.join([str(i) for i in output.shape]),
        # ))
        self.torch_ir.convert_function(func, inputs, output, *args, **kwargs)
        return True
