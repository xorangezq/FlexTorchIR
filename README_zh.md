# FlexTorchIR

([EN](README.md) | [ZH (本文档)](README_zh.md))

FlexTorchIR 是不同于 ONNX 和 `torch.trace` 的一种全新的 pytorch 模型转换方法。



本项目零依赖，不仅仅是指不依赖任何第三方库，更是指不依赖任何额外的概念和理解成本。它只使用 python 和 pytorch 来实现转换。



只要会写 pytorch 模型，就能够支持任何层、任意粒度的转换，哪怕是自定义的网络层也可以零额外成本顺利支持。就算这个 pytorch 网络无法导出 ONNX 格式，或者在 `torch.trace` 中会丢失另一些运行时的分支，都可以用本项目实现完整支持。



转换 AI 模型中会遇到很多固有的困难，在 ONNX 中这些困难被拆分成了众多细碎的 op 以及各种子版本，在 `torch.trace` 中则是利用一些前提限制条件排除。但凡想长久的支持一个持续发展的 AI 框架以及另一个持续发展的推理引擎，这些复杂度就始终会有。这些固有的复杂度并不会被消除，在本项目中同样存在，却是以另一种方式来表达。关于复杂度管理、设计原理等具体内容，请参阅 `DESIGN.md`。



没有一种 AI 模型转换的方法是完美的，不过，如果你遇到了下面的其中一种情况，`FlexTorchIR` 绝对能帮到你：

- 当我手头上有一个 pytorch 模型，我就想把它转换出来，又要快、又要准，完全不想被 ONNX，或者各种各样的转换工具和格式绊住手脚；
- 手上的模型有个自定义的层，要支持的话还需要在各种转换工具里提 issue、等着添加支持（或者自己来）；—— 用 `FlexTorchIR` 就可以直接写转换代码了
- `ONNX` 格式在开源社区中才有绝对的必要性，如果为闭源项目开发（公司也好其他甲方也好），完全只需要从 pytorch 模型直接转到自己定义的私有格式即可；
- `torch.trace` 确实设计的不错，也挺好用的，但除了从 torch 底层的角度“打印”一张模型运行时的快照，我们还可以就跟编写、运行一个 pytorch 模型一样，就在同样的 python 代码下，用 `FlexTorchIR` 直接逐层 hook 出来转换。



### Installation 安装

```
git clone ...
pip install -e .
```

推荐使用 `-e` ，因为在模型转换的同时，一般都需要同步修改 `FlexTorchIR` 的代码。



###  Example 示例

```bash
cd examples/
python example_resnet50.py
# or python example_torchvision.py
```

输出如下：



第一部分，支持的转换，包括：

- 层/算子 类型
- 层/算子 各项参数

这些信息足够用于构建你自己的私有 IR 层和算子：

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



第二部分，不支持的转换，包括：

- 不支持的层/算子 类型
- 该层的输入和输出 tensor shape

```
INFO     | [WARNING][TorchConverter] The following Ops are not supported by flexir currently: 
INFO     | Op: (func op) flatten. Input shapes: 1x2048x1x1, Output shape: 1x2048 
INFO     | [WARNING][TorchConverter] These Ops has been converted into LayerPlaceholder and are shown in color **RED** in model png. 
resnet50 success
```

`FlexTorchIR` 已经支持绝大部分常见的 CV 模型，但在转换实际部署的 AI 模型时，还是经常会遇到一些不支持的层。

别慌！转换已经成功，放轻松：

- `FlexTorchIR` 遇到这些层/算子，并不会出错、退出、异常。它会继续正常转换，就像没遇到错误一样；
- 所有不支持的层/算子，都会用 `LayerPlaceHolder` 类型来顶替。你可以在转换过后，一个个地添加支持；
- 你可以在转换过后，对“还需要支持多少算子”这个问题的答案一目了然。这是一个为你工作的工具，而不把你当作工具。你能很轻松的规划自己的工作量，或者调整模型的复杂度，增删哪些层也能一针见血。这样的工作流，就像是 OOP 依赖反转一样。



### FlexTorchIR 所不包含的

FlexTorchIR 不包含转换后的模型 IR：你需要自己实现。



`FlexTorchIR` 提供了一种转换 pytorch 模型的方法，也仅仅只提供了一个方法。诚然，提供转换却没有转换后的 IR，感觉事情只做了一半。但你最终不也得再转成自己实际部署的私有 IR 吗？正因为必须要提供一个转换结果 IR，ONNX 才有了那么多乱七八糟的没人爱用的内容，去兼容不断增长的 torch op 参数集，以及各种或细分又合并的 op 类型。



支持不断发展的 torch 算子、兼容每个私有项目中的二进制部署格式，这些本就是每个私有项目应该面临的复杂度，也只应存在于每个私有项目中自行管理。一个开源的模型转换方法，如果把这个复杂度绑在自己身上，非但没有简化问题，还会迫使每一个下游项目都要处理每一处的最大的复杂性。



如果一个小项目，做不成的话都没有后续了，它只需要集中精力把手上的模型转换出来，还考虑对 torch 版本的兼容性干嘛？转不出 ONNX 也没关系，`FlexTorchIR` 此时就可以帮到你。



这也是为什么，每个 AI 推理引擎都有自己的转换脚本，ncnn、MNN、ggml 等等。这都是必然的，也是应该的。但这也意味着你必须也同样使用他们的推理引擎来部署。



模型转换，只是一个函数，像一个管道一样，模型进、IR 出。



`FlexTorchIR` 提供了这样一个管道，保证能支持任何 pytorch 模型，设计非常直接，很容易就能新增、修改。至于转换出什么，交给你来定义。



TODO：

- [ ] 提供 protobuf 格式，作为默认的目标 IR

(请注意：任何目标 IR 的设计决定，都实际上对最终部署提出了一定的要求和假设。)



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

