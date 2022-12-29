import torch
import torch.nn as nn
from typing import Any, Union, Optional, Sequence, Mapping, Iterable, TypeVar
from torchinfo import summary

from .thop import profile, clever_format

INPUT_DATA_TYPE = Union[torch.Tensor, Sequence[Any], Mapping[str, Any]]
CORRECTED_INPUT_DATA_TYPE = Optional[Union[Iterable[Any], Mapping[Any, Any]]]
INPUT_SIZE_TYPE = Sequence[Union[int, Sequence[Any], torch.Size]]
DEVICE = TypeVar('DEVICE', torch.device, str)


def calc_flop(model, args, format='%.3f'):
    flops, params = profile(model, args)
    flops, params = clever_format([flops, params], format)

    return {'flops': flops, 'params': params}

def remove_gradient(model):
    for param in model.parameters():
        param.requires_grad = False

def vsummary(model: nn.Module,
    input_data: INPUT_DATA_TYPE = None,
    batch_dim: int = None,
    cache_forward_pass: bool = None,
    col_names: Iterable[str] = None,
    col_width: int = 25,
    depth: int = 3,
    device: DEVICE = None,
    mode: str = None,
    row_settings: Iterable[str] = None,
    verbose: int = None,
    **kwargs: Any,):

    sr = summary(
        model= model, 
        input_data = input_data, 
        batch_dim = batch_dim, 
        cache_forward_pass = cache_forward_pass, 
        col_names = col_names, 
        col_width = col_width, 
        depth = depth, 
        device = device, 
        mode = mode, 
        row_settings = row_settings, 
        verbose = verbose,
        **kwargs)

    remove_gradient(model)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(input_data, dict):
        for key in input_data.keys():
            if isinstance(input_data[key], torch.Tensor):
                input_data[key] = input_data[key].to(device)
    elif isinstance(input_data, (tuple, list)):
        if len(input_data) > 1:
            input_data = [data.to(device) for data in input_data]
        else:
            input_data = input_data[0].to(device)
    else:
        input_data = input_data.to(device)

    flops, params = profile(model, input_data, verbose=False)
    flops, params = clever_format([flops, params], '%.4f')
    print("Forward params (%s): " % (params[-1]), params[:-1])
    print("Total Flops (%s): " % (flops[-1]), flops[:-1])
    print('=' * sr.formatting.get_total_width())

    if isinstance(input_data, torch.Tensor):
        print("Input Shape: ", list(input_data.size()))
    elif isinstance(input_data, Sequence):
        for i, v in enumerate(input_data):
            if isinstance(v, torch.Tensor):
                print('Input Shape[%d]:' % (i), list(v.size()))
            else:
                print('Input Value[%d]:' % (i), v)
    elif isinstance(input_data, Mapping):
        for k in input_data.keys():
            if isinstance(input_data[k], torch.Tensor):
                print("Input Shape[%s]:" % (k), list(input_data[k].size()))
            else:
                print('Input Value[%s]:' % (k), input_data[k])
    print('=' * sr.formatting.get_total_width())