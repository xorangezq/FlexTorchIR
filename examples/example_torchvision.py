#!/usr/bin/env python
# coding: utf-8

import importlib

import torch
import torchvision

regression_torchvision_models = [
    ('resnet50',            torchvision.models.resnet50, (1, 3, 224, 224)),

    ('squeezenet1_0', torchvision.models.squeezenet1_0, (1, 3, 224, 224)),
    #
    # https://pytorch.org/vision/stable/models.html#classification
    #
    ('mobilenet_v2', torchvision.models.mobilenet_v2, (1, 3, 224, 224)),

    ('resnet18',            torchvision.models.resnet18, (1, 3, 224, 224)),
    ('resnet34',            torchvision.models.resnet34, (1, 3, 224, 224)),
    ('resnet50',            torchvision.models.resnet50, (1, 3, 224, 224)),
    ('resnet101',           torchvision.models.resnet101, (1, 3, 224, 224)),
    ('resnet152',           torchvision.models.resnet152, (1, 3, 224, 224)),
    ('resnext50_32x4d',     torchvision.models.resnext50_32x4d, (1, 3, 224, 224)),
    ('resnext101_32x8d',    torchvision.models.resnext101_32x8d, (1, 3, 224, 224)),
    ('wide_resnet50_2',     torchvision.models.wide_resnet50_2, (1, 3, 224, 224)),
    ('wide_resnet101_2',    torchvision.models.wide_resnet101_2, (1, 3, 224, 224)),

    ('alexnet',             torchvision.models.alexnet, (1, 3, 224, 224)),

    ('vgg11',       torchvision.models.vgg11, (1, 3, 224, 224)),
    ('vgg11_bn',    torchvision.models.vgg11_bn, (1, 3, 224, 224)),
    ('vgg13',       torchvision.models.vgg13, (1, 3, 224, 224)),
    ('vgg13_bn',    torchvision.models.vgg13_bn, (1, 3, 224, 224)),
    ('vgg16',       torchvision.models.vgg16, (1, 3, 224, 224)),
    ('vgg16_bn',    torchvision.models.vgg16_bn, (1, 3, 224, 224)),
    ('vgg19',       torchvision.models.vgg19, (1, 3, 224, 224)),
    ('vgg19_bn',    torchvision.models.vgg19_bn, (1, 3, 224, 224)),

    ('squeezenet1_0', torchvision.models.squeezenet1_0, (1, 3, 224, 224)),
    ('squeezenet1_1', torchvision.models.squeezenet1_1, (1, 3, 224, 224)),

    ('densenet121', torchvision.models.densenet121, (1, 3, 224, 224)),
    ('densenet161', torchvision.models.densenet161, (1, 3, 224, 224)),
    ('densenet169', torchvision.models.densenet169, (1, 3, 224, 224)),
    ('densenet201', torchvision.models.densenet201, (1, 3, 224, 224)),

    ('inception_v3', torchvision.models.inception_v3, (1, 3, 299, 299)),

    ('googlenet', torchvision.models.googlenet, (1, 3, 224, 224)),

    ('shufflenet_v2_x0_5', torchvision.models.shufflenet_v2_x0_5, (1, 3, 224, 224)),
    ('shufflenet_v2_x1_0', torchvision.models.shufflenet_v2_x1_0, (1, 3, 224, 224)),
    ('shufflenet_v2_x1_5', torchvision.models.shufflenet_v2_x1_5, (1, 3, 224, 224)),
    ('shufflenet_v2_x2_0', torchvision.models.shufflenet_v2_x2_0, (1, 3, 224, 224)),

    ('mobilenet_v2', torchvision.models.mobilenet_v2, (1, 3, 224, 224)),
    # ('mobilenet_v3_large', torchvision.models.mobilenet_v3_large, (1, 3, 224, 224)),
    # ('mobilenet_v3_small', torchvision.models.mobilenet_v3_small, (1, 3, 224, 224)),

    ('mnasnet0_5',  torchvision.models.mnasnet0_5, (1, 3, 224, 224)),
    ('mnasnet0_75', torchvision.models.mnasnet0_75, (1, 3, 224, 224)),
    ('mnasnet1_0',  torchvision.models.mnasnet1_0, (1, 3, 224, 224)),
    ('mnasnet1_3',  torchvision.models.mnasnet1_3, (1, 3, 224, 224)),
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
            break
        # break

    for name,result in results.items():
        print(name, result)

'''
resnet50 success
squeezenet1_0 success
mobilenet_v2 success
resnet18 success
resnet34 success
resnet101 success
resnet152 success
resnext50_32x4d success
resnext101_32x8d success
wide_resnet50_2 success
wide_resnet101_2 success
alexnet success
vgg11 success
vgg11_bn success
vgg13 success
vgg13_bn success
vgg16 success
vgg16_bn success
vgg19 success
vgg19_bn success
squeezenet1_1 success
densenet121 success
densenet161 success
densenet169 success
densenet201 success
inception_v3 success
googlenet success
shufflenet_v2_x0_5 success
shufflenet_v2_x1_0 success
shufflenet_v2_x1_5 success
shufflenet_v2_x2_0 success
mnasnet0_5 success
mnasnet0_75 success
mnasnet1_0 success
mnasnet1_3 success
'''
