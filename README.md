# mtools
calc model‘s information and convert model

## How To Use
```python
import torch
from mtools import vsummary
from torchvision.models import resnet50

if __name__ == '__main__':
    model = resnet50()
    input = torch.randn((1, 3, 224, 224))
    vsummary(model, input_data=(input,), depth=2)

# show
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   [1, 1000]                 --
├─Conv2d: 1-1                            [1, 64, 112, 112]         9,408
├─BatchNorm2d: 1-2                       [1, 64, 112, 112]         128
├─ReLU: 1-3                              [1, 64, 112, 112]         --
├─MaxPool2d: 1-4                         [1, 64, 56, 56]           --
├─Sequential: 1-5                        [1, 256, 56, 56]          --
│    └─Bottleneck: 2-1                   [1, 256, 56, 56]          75,008
│    └─Bottleneck: 2-2                   [1, 256, 56, 56]          70,400
│    └─Bottleneck: 2-3                   [1, 256, 56, 56]          70,400
├─Sequential: 1-6                        [1, 512, 28, 28]          --
│    └─Bottleneck: 2-4                   [1, 512, 28, 28]          379,392
│    └─Bottleneck: 2-5                   [1, 512, 28, 28]          280,064
│    └─Bottleneck: 2-6                   [1, 512, 28, 28]          280,064
│    └─Bottleneck: 2-7                   [1, 512, 28, 28]          280,064
├─Sequential: 1-7                        [1, 1024, 14, 14]         --
│    └─Bottleneck: 2-8                   [1, 1024, 14, 14]         1,512,448
│    └─Bottleneck: 2-9                   [1, 1024, 14, 14]         1,117,184
│    └─Bottleneck: 2-10                  [1, 1024, 14, 14]         1,117,184
│    └─Bottleneck: 2-11                  [1, 1024, 14, 14]         1,117,184
│    └─Bottleneck: 2-12                  [1, 1024, 14, 14]         1,117,184
│    └─Bottleneck: 2-13                  [1, 1024, 14, 14]         1,117,184
├─Sequential: 1-8                        [1, 2048, 7, 7]           --
│    └─Bottleneck: 2-14                  [1, 2048, 7, 7]           6,039,552
│    └─Bottleneck: 2-15                  [1, 2048, 7, 7]           4,462,592
│    └─Bottleneck: 2-16                  [1, 2048, 7, 7]           4,462,592
├─AdaptiveAvgPool2d: 1-9                 [1, 2048, 1, 1]           --
├─Linear: 1-10                           [1, 1000]                 2,049,000
==========================================================================================
Total params: 25,557,032
Trainable params: 25,557,032
Non-trainable params: 0
Total mult-adds (G): 4.09
==========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 177.83
Params size (MB): 102.23
Estimated Total Size (MB): 280.66
==========================================================================================
Forward params (M):  25.5570
Total Flops (G):  4.1337
==========================================================================================
Input Shape:  [1, 3, 224, 224]
==========================================================================================
```

### Document
```python
def vsummary(
    model: nn.Module,
    input_size: Optional[INPUT_SIZE_TYPE] = None,
    batch_dim: Optional[int] = None,
    cache_forward_pass: Optional[bool] = None,
    col_names: Optional[Iterable[str]] = None,
    col_width: int = 25,
    depth: int = 3,
    device: Optional[torch.device] = None,
    mode: str | None = None,
    row_settings: Optional[Iterable[str]] = None,
    verbose: int = 1,
    **kwargs: Any,
) -> ModelStatistics:
"""
Summarize the given PyTorch model. Summarized information includes:
    1) Layer names,
    2) input/output shapes,
    3) kernel shape,
    4) # of parameters,
    5) # of operations (Mult-Adds),
    6) whether layer is trainable

NOTE: If neither input_data or input_size are provided, no forward pass through the
network is performed, and the provided model information is limited to layer names.

Args:
    model (nn.Module):
            PyTorch model to summarize. The model should be fully in either train()
            or eval() mode. If layers are not all in the same mode, running summary
            may have side effects on batchnorm or dropout statistics. If you
            encounter an issue with this, please open a GitHub issue.

    input_data (Sequence of Tensors):
            Arguments for the model's forward pass (dtypes inferred).
            If the forward() function takes several parameters, pass in a list of
            args or a dict of kwargs (if your forward() function takes in a dict
            as its only argument, wrap it in a list).
            Default: None

    batch_dim (int):
            Batch_dimension of input data. If batch_dim is None, assume
            input_data / input_size contains the batch dimension, which is used
            in all calculations. Else, expand all tensors to contain the batch_dim.
            Specifying batch_dim can be an runtime optimization, since if batch_dim
            is specified, torchinfo uses a batch size of 1 for the forward pass.
            Default: None

    cache_forward_pass (bool):
            If True, cache the run of the forward() function using the model
            class name as the key. If the forward pass is an expensive operation,
            this can make it easier to modify the formatting of your model
            summary, e.g. changing the depth or enabled column types, especially
            in Jupyter Notebooks.
            WARNING: Modifying the model architecture or input data/input size when
            this feature is enabled does not invalidate the cache or re-run the
            forward pass, and can cause incorrect summaries as a result.
            Default: False

    col_names (Iterable[str]):
            Specify which columns to show in the output. Currently supported: (
                "input_size",
                "output_size",
                "num_params",
                "params_percent",
                "kernel_size",
                "mult_adds",
                "trainable",
            )
            Default: ("output_size", "num_params")
            If input_data / input_size are not provided, only "num_params" is used.

    col_width (int):
            Width of each column.
            Default: 25

    depth (int):
            Depth of nested layers to display (e.g. Sequentials).
            Nested layers below this depth will not be displayed in the summary.
            Default: 3

    device (torch.Device):
            Uses this torch device for model and input_data.
            If not specified, uses result of torch.cuda.is_available().
            Default: None

    mode (str)
            Either "train" or "eval", which determines whether we call
            model.train() or model.eval() before calling summary().
            Default: "eval".

    row_settings (Iterable[str]):
            Specify which features to show in a row. Currently supported: (
                "ascii_only",
                "depth",
                "var_names",
            )
            Default: ("depth",)

    verbose (int):
            0 (quiet): No output
            1 (default): Print model summary
            2 (verbose): Show weight and bias layers in full detail
            Default: 1
            If using a Juypter Notebook or Google Colab, the default is 0.

    **kwargs:
            Other arguments used in `model.forward` function. Passing *args is no
            longer supported.

Return:
    ModelStatistics object
            See torchinfo/model_statistics.py for more information.
"""

### Reference

- [x] https://github.com/TylerYep/torchinfo
- [x] https://github.com/Lyken17/pytorch-OpCounter
```