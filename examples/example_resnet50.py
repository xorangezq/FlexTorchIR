#!/usr/bin/env python
# coding: utf-8

import importlib

import torch
import torchvision

regression_torchvision_models = [
    ('resnet50',            torchvision.models.resnet50, (1, 3, 224, 224)),
]

if __name__ == '__main__':

    results = {}
    def log_result(name, result):
        results[name] = result
    for name, module, shape in regression_torchvision_models:
        print('Converting %s ...' % name)
        inputs = [torch.randn(*shape)]
        torch_model = module()
        # torch_model = module(pretrained=True)
        # import pdb; pdb.set_trace()

        try:
            from flexir.tools.converter import convert_pytorch_model
            convert_pytorch_model(
                torch_model,
                inputs,
                modelname=name,
                outputfolder='flexir_convert_demonstrate/',
            )
            log_result(name, 'success')
        except Exception as e:
            log_result(name, 'failed')
            raise e

    for name,result in results.items():
        print(name, result)
