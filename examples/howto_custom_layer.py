#!/usr/bin/env python
# coding: utf-8

'''
Examples for: How to support conversions for custom layers?
1. Softargmax (https://github.com/david-wb/softargmax)
2. Any custom module
'''

import numpy as np
import torch

class SoftArgMax(torch.nn.Module):
    '''
    Step 1: translate to a torch.nn.Module
    Step 2: implement it, use any torch mod/func as you need
    Step 3: Add conversion in flexir.graph.torch.convert:AVALIABLE_MODULE_CONVERTER
            already did for SoftArgMax example.
    And it's OK!

    Also noted: any thing used in custom module would not trigger another hook !
    This means: you may wrap anything inside a custom layer, and implement a dedicated compute kernel for it.
    '''
    def __init__(self):
        super(SoftArgMax, self).__init__()
        self.beta = 100

    def forward(self, input):
        '''
        https://github.com/david-wb/softargmax/blob/master/softargmax.py
        '''
        beta = self.beta

        *_, h, w = input.shape

        input = input.reshape(*_, h * w)
        input = torch.nn.functional.softmax(beta * input, dim=-1)

        indices_c, indices_r = np.meshgrid(
            np.linspace(0, 1, w),
            np.linspace(0, 1, h),
            indexing='xy'
        )

        indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w)))
        indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w)))

        result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
        result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

        result = torch.stack([result_r, result_c], dim=-1)

        return result

class ExampleDriver(torch.nn.Module):
    def __init__(self):
        super(ExampleDriver, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, 3)
        self.softargmax = SoftArgMax()

    def forward(self, x):
        y = self.conv(x)
        y = torch.nn.functional.relu(y)
        y = self.softargmax(y)
        return y

demo = ExampleDriver()
y = demo(torch.randn(1, 3, 224, 224))

from flexir.tools.converter import convert_pytorch_model
convert_pytorch_model(
    demo,
    [torch.randn(1, 3, 224, 224)],
    modelname='howto_demo',
    outputfolder='flexir_convert_demonstrate/',
)

'''
Output:
DEBUG    | Converting... Conv2d, C3 N3 K3x3 S3x3 P00 D1x1 G1 hasBias | flexir.graph.torch.convert.common:convert_Module_Conv2d:212 |
DEBUG    | Converting... ReLU | flexir.graph.torch.convert.activation:convert_torch_nn_functional_relu:11 |
INFO     | torch.stack | flexir.graph.torch.hook:hook_fn:229 |
DEBUG    | Converting... SoftArgmax, beta 100 | flexir.graph.torch.convert.common:convert_Module_SoftArgMax:474 |
INFO     | Model diagram is generated at flexir_convert_demonstrate/howto_demo/howto_demo.png | flexir.graph.network:save:54 |
'''
