# FlexTorchIR

([EN (this doc)](README.md) | [ZH](README_zh.md))

FlexTorchIR is a new way to convert pytorch models into your own proprietary IR, without ONNX or `torch.trace`. 



It is zero-dependence, not only to third-party libraries, but also to any extra concept other than pytorch itself. It use merely python and pytorch to convert pytorch models.



That means, if a pytorch model works, it's guaranteed to be able to convert to fully customizable proprietary IR, even though it may not be able to convert to ONNX, or may be partially limited by `torch.trace` in runtime control flow.



Accidental complex problems always lie, either incartenated as ONNX ops and version blackhole, or one agree to `torch.trace`'s limitation(or assumption)(or prerequisites). `FlexTorchIR` also carries the same burden though in a different way. For more details, please refer to `DESIGN.md`, which also explains how it works.



No AI model conversion is perfect, that said, `FlexTorchIR` could be really helpful if:

- You just want to get the work done and refuse to be distracted away by any other complex.
- You're working closely with the model team, and guarantee to convert any model they produce. Or you design the AI model yourself.
- `ONNX` is only a standard way for open source projects. If you work proprietarily, you don't need it at all.
- `torch.trace` is great I agreed. Instead of tracing the model in the lower part of the architecture , you may trace it directly at the level you write it, with `FlexTorchIR`.



### Installation

```bash
git clone ... && cd ...
pip install -e .
```

It is recommended to use `-e` so that one could actively develop converted module support along with converting a new pytorch model.



`FlexTorchIR` is expected to be in actively development while converting a new pytorch model.



### Example

```bash
cd examples/
python example_resnet50.py
# or python example_torchvision.py
```

Outputs:



First part, recognized (registered) conversion:

- layer type
- layer params

You may store these necessities into your own proprietary IR layers.

```
Converting resnet50 ...
DEBUG    | Converting... Conv2d, C3 N64 K7x7 S2x2 P33 D1x1 G1  
DEBUG    | Converting... BatchNorm2d, N 64 eps 1e-05 momentum 0.1 affine track_stats 
DEBUG    | Converting... ReLU 
DEBUG    | Converting... MaxPool2d, K3x3 S2x2 P11 D1x1 ceil?N 
DEBUG    | Converting... Conv2d, C64 N64 K1x1 S1x1 P00 D1x1 G1  
DEBUG    | Converting... BatchNorm2d, N 64 eps 1e-05 momentum 0.1 affine track_stats 
DEBUG    | Converting... ReLU 
( ... Omitting for readability ... )
DEBUG    | Converting... Conv2d, C2048 N512 K1x1 S1x1 P00 D1x1 G1  
DEBUG    | Converting... BatchNorm2d, N 512 eps 1e-05 momentum 0.1 affine track_stats 
DEBUG    | Converting... ReLU 
DEBUG    | Converting... Conv2d, C512 N512 K3x3 S1x1 P11 D1x1 G1  
DEBUG    | Converting... BatchNorm2d, N 512 eps 1e-05 momentum 0.1 affine track_stats 
DEBUG    | Converting... ReLU 
DEBUG    | Converting... Conv2d, C512 N2048 K1x1 S1x1 P00 D1x1 G1  
DEBUG    | Converting... BatchNorm2d, N 2048 eps 1e-05 momentum 0.1 affine track_stats 
DEBUG    | Converting... ReLU 
DEBUG    | Converting... AvgPool2d (from Adaptive 1x1), K7x7 S1x1 P00 D1x1 ceil?N countIncludePad 
INFO     | catched unrecognized func flatten 
DEBUG    | Converting... FullConnection, C2048 N1000 
```



Second Part, unrecognized conversion:

- layer type (that's unrecognized (unregistered))
- input shape and output shape of the unrecognized layer

```
INFO     | [WARNING][TorchConverter] The following Ops are not supported by flexir currently: 
INFO     | Op: (func op) flatten. Input shapes: 1x2048x1x1, Output shape: 1x2048 
INFO     | [WARNING][TorchConverter] These Ops has been converted into LayerPlaceholder and are shown in color **RED** in model png. 
resnet50 success
```

Although `FlexTorchIR` already supports most reasonable models in CV, It is common that when converting a new model in production, some layers in use are not in support list.

Don't panic ! Relax because:

- `FlexTorchIR` will not abort, fail or reject these layers. Instead, it continues conversion as if everything is fine.
- All unsupported layers would be mark with a `LayerPlaceHolder` type. You may work on it layer, one at a time.
- You would figure out how much work to support UP FRONT. You are the master of the workflow, not a slave to it. Think of it similar to a dependency inversion.



### What FlexTorchIR does not have

FlexTorchIR does not have the proprietary IR definition. You'll have to write your own.



`FlexTorchIR` offers a way to convert pytorch models. But a conversion without target IR is only half complete. This is why every conversion tool offers a new set of IRs for AI model, which inevitably breeds complex and confusion for evolving parameter sets, incompatibilities,  and dilemma of finer ops for better support or coarser ops for better understanding, etc.



These complex, supporting AI framework's evolving parameter sets, providing compatibilities across model and binary executable's versions, by its very nature, lie in production engineering and proprietary IR. And the new sets of IRs provided by conversion tools, not only did not simplify the problem, but every client project is also forced to deal with the same problem, in a manner of highest possible complicated level.



This insight also holds that every AI inference engine provides their own conversion tool, ncnn, MNN, ggml. This is perfectly fine, except that you'll have to use their engine to infer in production.



Conversion is a function. It is a pipe that takes in AI model and produce your IR.



`FlexTorchIR` provides you this pipe, guarantees to work for all pytorch model, and is straightforward to develop and modify, in the expense that, You'll have to provide your own proprietary IR.



TODO:

- [ ] provide protobuf format as a default IR for conversion

(Please noted: Any decision made on default IR more or less and almost surely assumed something for your production.)



### License

```
MIT License

Copyright (c) 2025 oliverxu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

