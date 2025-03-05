#!/usr/bin/env python
# coding: utf-8
import os

import torch

import flexir.graph.torch.convert as convert
from flexir.graph.layers import LayerPlaceholder

from flexir.utilities.logger import logger, ASSERT

class PytorchNodeInfo(object):
    def __init__(self, kernel=None, inputs=[], outputs=[]):
        super(PytorchNodeInfo, self).__init__()
        # pytorch origin objects, will delete after reconstruction.
        self._func = kernel
        self._inputs = inputs
        self._outputs = outputs
        self._if_is_output = False
        self._output_loc = -1
        self._converted_layer = None
        pass

    def __repr__(self):
        return self._converted_layer.__repr__()

    def __str__(self):
        return self._converted_layer.__str__()

    def convert_Function_Info(self, *args, **kwargs):
        try:
            if self._func in convert.AVALIABLE_FUNCTION_CONVERTER.keys():
                self._converted_layer = convert.AVALIABLE_FUNCTION_CONVERTER[self._func](self._func, self._inputs, self._outputs, *args, **kwargs)
                # for any branches of implementation, you may also throw NotImplementedError to use LayerPlaceholderInfo as conversion fallback
            else:
                raise NotImplementedError('{funcname} not found in AVALIABLE_FUNCTION_CONVERTER'.format(funcname=self._func.__name__))
        except NotImplementedError:
            logger.info('catched unrecognized func {}', self._func.__name__)
            self._converted_layer = convert.convert_torch_unrecognized('(func op) ' + self._func.__name__, self._inputs, self._outputs, *args, **kwargs)

        self._converted_layer._id = TorchIR.tensorID(self._outputs)
        self._converted_layer._read_count = 1
        return True

    def convert_Module_Info(self):
        ASSERT(isinstance(self._func, torch.nn.Module), '''self.func[{}] is not a module'''.format(self._func.__class__.__name__))

        try:
            if self._func.__class__.__name__ in convert.AVALIABLE_MODULE_CONVERTER.keys():
                self._converted_layer = convert.AVALIABLE_MODULE_CONVERTER[self._func.__class__.__name__](self._func, self._inputs, self._outputs)
                # for any branches of implementation, you may also throw NotImplementedError to use LayerPlaceholderInfo as conversion fallback
            else:
                raise NotImplementedError('{modulename} not found in AVALIABLE_MODULE_CONVERTER'.format(modulename=self._func.__class__.__name__))
        except NotImplementedError:
            logger.info('catched unrecognized mod {}', self._func.__class__.__name__)
            self._converted_layer = convert.convert_torch_unrecognized('(module) ' + self._func.__class__.__name__, self._inputs, self._outputs)

        ASSERT(self._converted_layer is not None, 'Fail to convert module<{}>'.format(self._func.__class__.__name__))
        self._converted_layer._input_ids = [TorchIR.tensorID(inp) for inp in convert.recurrent_split_arguments(self._inputs)]
        if self._func.__class__.__name__ in ['RNN', 'LSTM', 'GRU', 'LSTMCell', 'RNNCell', 'GRUCell']:
            self._converted_layer._input_ids = [TorchIR.tensorID(self._inputs[0])]

        self._converted_layer._id = TorchIR.tensorID(self._outputs)
        self._converted_layer._read_count = 1
        return True


class TorchIR:

    def __init__(self):
        '''
        layer ID binds to (its output) blob ID
        '''
        self.layers = {}
        self.blobs = {}

    @staticmethod
    def tensorID(tensor):
        return str(id(tensor))

    def convert_module(self, module, inputs, output):
        PyLayer = PytorchNodeInfo(module, inputs, output)
        rt = PyLayer.convert_Module_Info()
        ASSERT(rt, 'convert_Module_Info Failed.')

        if id(output) == id(inputs[0]) and not isinstance(PyLayer._converted_layer, IdentityInfo):
            ASSERT(False, 'do not support inplace layer.')

        if PyLayer._converted_layer._id not in self.layers.keys():
            self.layers[PyLayer._converted_layer._id] = PyLayer

            for _input in inputs:
                self.blobs[TorchIR.tensorID(_input)] = _input
            self.blobs[TorchIR.tensorID(output)] = output
        else:
            ASSERT(isinstance(PyLayer._converted_layer, IdentityInfo))

    def convert_function(self, func, inputs, output, *args, **kwargs):
        PyLayer = PytorchNodeInfo(func, inputs, output)
        rt = PyLayer.convert_Function_Info(*args, **kwargs)
        if rt:
            self.layers[TorchIR.tensorID(output)] = PyLayer

            for _input in inputs:
                self.blobs[TorchIR.tensorID(_input)] = _input
            self.blobs[TorchIR.tensorID(output)] = output

    def post_convert(self):
        self.setPytorchNodeInfosReadcount()

        self.warnAnyUnsupportedLayers()

    def setPytorchNodeInfosReadcount(self):
        for key in self.layers.keys():
            if self.layers[key]._converted_layer is None:
                continue
            self.layers[key]._converted_layer._read_count = 0
        for key in self.layers.keys():
            if self.layers[key]._converted_layer is None:
                continue
            layer = self.layers[key]
            for in_key in layer._converted_layer._input_ids:
                if in_key in self.layers.keys():
                    self.layers[in_key]._converted_layer._read_count += 1

    def warnAnyUnsupportedLayers(self):
        unsupported_layers_count = 0
        for l in self.layers.values():
            if isinstance(l._converted_layer, LayerPlaceholder):
                unsupported_layers_count += 1

        if unsupported_layers_count != 0:
            logger.info('[WARNING][TorchConverter] The following Ops are not supported by flexir currently:')

            format_shape = lambda shape: 'x'.join([str(s) for s in shape])

            for l in self.layers.values():
                layer_info = l._converted_layer
                if isinstance(layer_info, LayerPlaceholder):
                    logger.info('{opdesc}. Input shapes: {ishapes}, Output shape: {oshape}'.format(
                        opdesc=layer_info._desc,
                        ishapes=', '.join([format_shape(shape) for shape in layer_info._input_shapes]),
                        oshape=format_shape(layer_info._output_shape),
                    ))
            logger.info('[WARNING][TorchConverter] These Ops has been converted into LayerPlaceholder and are shown in color **RED** in model png.')

    def set_outputs(self, outputs):
        self.setPytorchNodeInfosIsOutputs(output_ids_list=[TorchIR.tensorID(outp) for outp in outputs])

    def setPytorchNodeInfosIsOutputs(self, output_ids_list=[]):
        output_loc = 0
        for key in self.layers.keys():
            if key in output_ids_list:
                self.layers[key]._if_is_output = True
                self.layers[key]._output_loc = output_ids_list.index(key)
                continue
            if self.layers[key]._converted_layer is None:
                continue
            if self.layers[key]._converted_layer._read_count == 0:
                self.layers[key]._if_is_output = True
                self.layers[key]._output_loc = len(output_ids_list)+output_loc
                output_loc += 1

    def buildNetwork(self, ifDraw=True, drawPath='./', device='cpu'):
        from collections import OrderedDict
        from flexir.graph.network import NetGraph

        G = NetGraph()

        for layer_id, layer in self.layers.items():
            node_desc = 'ID({0})\nShape{1}'.format(layer_id, tuple(layer._outputs.size()))
            G.add_node(str(layer_id), node_desc)

        for layer_id in self.layers.keys():
            for in_layer_id in self.layers[layer_id]._converted_layer._input_ids:
                if in_layer_id not in self.layers.keys():
                    G.add_node(str(in_layer_id), 'Input[' + str(in_layer_id) + ']')

                G.add_edge(str(in_layer_id), str(layer_id))
            # end-for
        # end-for
        if ifDraw:
            try:
                G.GViz.render(os.path.join(drawPath, 'net_node_shapes'), cleanup=True)
            except Exception as e:
                logger.info('[EXCEPTION] {}', e)
                logger.info('[WARNING][buildNetwork] Install graphviz dot command to draw png.')

        sorted_nodes = G.topologicalsort()

        return G

    def optimize(self, ifMergeBN=True):
        '''
        Implement all kinds of optimization techniques here
        anything that modifies the network

        for instance:
        merge BatchNorm
        Channel Pruning
        etc ...
        '''
        if ifMergeBN:
            self.mergeBatchNorm2dIntoConv2d()
            self.mergeBatchNorm2dIntoConvTranspose2d()
            self.mergeBatchNorm1dIntoFullConnection()

        self.mergeWarpAffine()
        # ...

    def mergeBatchNorm2dIntoConv2d(self):
        pass
    def mergeBatchNorm2dIntoConvTranspose2d(self):
        pass
    def mergeBatchNorm1dIntoFullConnection(self):
        pass

    def mergeWarpAffine(self):
        '''
        merges gridSample + affineGrid into a single warpAffine param-set
        '''
        pass
