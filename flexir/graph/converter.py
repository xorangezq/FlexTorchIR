#!/usr/bin/env python
# coding: utf-8

from contextlib import contextmanager
from functools import reduce

import torch

from flexir.graph.torch.hook import ConvertContext as TorchContext
from flexir.graph.torch.ir import TorchIR
from flexir.utilities.logger import logger, ASSERT

class TorchConverter:
    '''
    This class converts torch modules into flexir model

    Usage:
    ```
    converter = TorchConverter()
    with converter.hook(module):
        module.eval()
        with torch.no_grad():
            output = module(*inputs)
    fir_model = converter.build_model(inputs, output)

    # converter.context is responsible for hooking and bookkeeping
    # Does NOT retain any state when converting is finished
    '''

    def __init__(self, output_path):
        self.context = TorchContext()
        self.output_path = output_path

    @contextmanager
    def hook(self, module):
        try:
            self.torch_ir = TorchIR()
            self.context.setup(module, self.torch_ir)
            yield
        finally:
            self.context.teardown()
            self.torch_ir.post_convert()

    def build_model(self, inputs, outputs, **kwargs):
        '''
        Arg:
            inputs, actual inputs for torch model, may be Tensor or List
            outputs, outputs of interest, may be Tensor or List
            (inputs and outputs must be those used within hook() context)

            kwargs and defaults:
                ifDraw = False,
                drawPath = './',
                device = 'cpu'
        '''
        inputs = dfs_flatten_to_list(inputs)
        outputs = dfs_flatten_to_list(outputs)

        abortText = lambda title, idx : \
            'Error. Tensor {}[{}] not recognized. \
Must assign tensors used in hook().'.format(title, idx)
        [ASSERT(TorchIR.tensorID(_input) in self.torch_ir.blobs.keys(), abortText('inputs', i)) for i,_input in enumerate(inputs)]
        [ASSERT(TorchIR.tensorID(output) in self.torch_ir.blobs.keys(), abortText('outputs', i)) for i,output in enumerate(outputs)]

        self.torch_ir.set_outputs(outputs)

        self.torch_ir.optimize()

        network = self.torch_ir.buildNetwork( **kwargs )
        return network

def dfs_flatten_to_list(item):
    if isinstance(item, torch.Tensor):
        return [item]
    assert(isinstance(item, (tuple, list)))
    return reduce(lambda returnlist,x : returnlist+dfs_flatten_to_list(x), item, [])
