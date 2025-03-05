#!/usr/bin/env python
# coding: utf-8

import os
import json

import torch

from flexir.graph.converter import TorchConverter

def mkdir_if_not_found(pathdir):
    if not os.path.exists(pathdir):
        import pathlib
        pathlib.Path(pathdir).mkdir(parents=True)
    elif os.path.isfile(pathdir):
        raise TypeError('\'%s\' is expected to be a directory but is instead a file.' % path)

def convert_pytorch_model(
    module,
    inputs,
    modelname,
    outputfolder,
    encrypt=True,
    **kwargs
):
    '''
    This function converts a pytorch model to flexIR (customizable at fine level) model, which is serialized in outputfolder/modelname/

    Args:
        module, the pytorch model to convert
        inputs, list of NCHW torch.tensors as inputs for dynamic module
        modelname, which is used to describe several outputs
        outputfolder, the folder to contain output

    Return:
        fir.graph.ir.FIRModel

    Output:
        All would be serialized in folder "outputfolder/modelname/", which would be created if not exists.
        Outputs include:
            ${modelname}_model.json, the converted fir model
            ${modelname}_graph.png, visualized fir model
    '''
    if not isinstance(inputs, (list, tuple)):
        raise TypeError('inputs are expected in list (of NCHW torch.tensors) but instead is %s' % str(type(inputs)))
    elif not isinstance(inputs[0], torch.Tensor):
        raise TypeError('inputs are expected in list of NCHW torch.tensors but instead is list of %s' % str(type(inputs[0])))

    output_path = os.path.join(outputfolder, modelname)
    mkdir_if_not_found(output_path)

    converter = TorchConverter(output_path)

    with converter.hook(module):
        module.eval()
        with torch.no_grad():
            output = module(*inputs, **kwargs)

    fir_model = converter.build_model(inputs, output)
    fir_model.save(name=modelname, outputDir=output_path)

    return fir_model
