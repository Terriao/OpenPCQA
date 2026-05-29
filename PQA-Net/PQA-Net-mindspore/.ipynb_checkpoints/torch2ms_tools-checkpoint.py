# -*- coding: utf-8 -*-
import mindspore as ms
import mindspore.ops as ops
from mindspore.common._register_for_tensor import tensor_operator_registry


def fill(input, val):
    if not isinstance(val, (int, float, bool)):
            raise TypeError("For 'Tensor.fill', the type of the argument 'value' must be int, float or bool, "
                            "but got {}.".format(type(val)))
    output = tensor_operator_registry.get("fill")(input.dtype, input.shape, val)
    return output

def uniform(input, a, b):
    return ms.Tensor(ms.common.initializer._init_random_uniform(a, b, input.shape), dtype=input.dtype)

def zeros(input):
    return tensor_operator_registry.get("fill")(input.dtype, input.shape, 0.0)

def repeat(inputx, *sizes):
    if isinstance(sizes[0], (tuple, list)):
        output = ms.ops.tile(inputx, *sizes)
    else:
        output = ms.ops.tile(inputx, sizes)
    return output

def clamp(input, min=None, max=None, out=None):
    type = input.dtype
    if min is not None and max is not None and min > max:
        output = ms.ops.ones_like(input).astype(type)*max
    else:
        if min is not None:
            min = ms.Tensor(min, type)
        if max is not None:
            max = ms.Tensor(max, type)
        output = ms.ops.clip_by_value(input, min, max)
    return output

def sigmoid(input):
    return 1 / (ms.ops.exp(0 - input) + 1)


def expand(input_ms, *size, is_under_gpu_context=True):
    @constexpr
    def size_to_ms_tensor(size):
        if isinstance(size[0], (list, tuple)):
            size = ms.Tensor(size[0])
        else:
            size = ms.Tensor(size)
        return size
    
    _size = size_to_ms_tensor(size)
    # TODO: ms.ops.expand() to support on GPU and delete 'broadcast_to' code.
    if is_under_gpu_context():
        return ms.ops.broadcast_to(input_ms, size)
    return input_ms.expand(_size)

def max(input, dim=None, keepdim=False):
    if dim is None:
        return input.max()

    indices, result = P.max(input, axis=dim, keep_dims=keepdim)
    return result, indices

# def tensor_bool_select(tensor_ms, index):
#     ms_shape_len = len(tensor_ms.shape)
#     index_shape_len = len(index.shape)
#     out_shape = [-1]
#     while index_shape_len < ms_shape_len:
#         out_shape.append(tensor_ms.shape[index_shape_len])
#         index = index.expand_dims(-1)
#         index_shape_len += 1
#     out = ms.ops.masked_select(tensor_ms, index)
#     if len(out_shape) > 1:
#         out = out.reshape(out_shape)

#     return out

def linspace(start, end, steps, dtype=None):
    if dtype is None:
        dtype = ms.float32
    start = ms.Tensor(start, dtype)
    end = ms.Tensor(end, dtype)
    output = ms.ops.linspace(start, end, steps)
    return output


def pad(input, pad, mode="constant", value=0):
    if mode == "replicate":
        mode = "edge"

    value = ms.Tensor(value, dtype=input.dtype)
    dims = len(input.shape)
    list_pad = [pad[i:i+2] for i in range(0, len(pad), 2)]
    list_pad.reverse()
    new_pad = [[0,0],] * int((dims - len(pad) /2))
    new_pad.extend(list_pad)

    @cast_tensor
    def _call_ms_api(input):
        # TODO: -> ms.ops.PadV3
        return ms.ops.operations.nn_ops.PadV3(mode=mode)(input, pad, value)

    outputs = _call_ms_api(input)
    return outputs


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):

    if return_indices is True or dilation != 1:
        raise NotImplementedError("These parameters cannot be set now.")

    _kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

    if stride is None:
        _stride = _kernel_size
    else:
        _stride = stride if isinstance(stride, tuple) else (stride, stride)

    _padding = padding if isinstance(padding, tuple) else (padding, padding)
    _dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    _extra_pad_h = 0
    _extra_pad_w = 0

    if ceil_mode:
        _input_shape = ms.ops.shape(input)
        _valid_in_h = (_input_shape[2] + _padding[0] * 2 - _dilation[0] * (_kernel_size[0] - 1) - 1)
        _valid_out_h =  _valid_in_h // _stride[0]
        tmp = _valid_out_h * _stride[0]
        if  tmp < _valid_in_h:
            _extra_pad_h = tmp + _stride[0] - _valid_in_h

        _valid_in_w = (_input_shape[3] + _padding[1] * 2 - _dilation[1] * (_kernel_size[1] - 1) - 1)
        _valid_out_w = _valid_in_w // _stride[1]
        tmp = _valid_out_w * _stride[1]
        if  tmp < _valid_in_w:
            _extra_pad_w = tmp + _stride[1] - _valid_in_w

    _pad = (((0, 0), (0, 0), (_padding[0], _padding[0] + _extra_pad_h), (_padding[1], _padding[1] + _extra_pad_w)))
    _max_pool = ms.ops.MaxPool(kernel_size=_kernel_size, strides=_stride, pad_mode='valid')

    cast_to_ms_tensor(input)
    # TODO: to support `value=float("inf")` in ms.ops.pad in future version
    input = ms.ops.pad(input, _pad)
    out = _max_pool(input)
    return cast_to_adapter_tensor(out)

def copy(input, src):
    output = ms.ops.broadcast_to(src, input.shape)
    output = output.astype(input.dtype)
    return output


def chunk(input, chunks, dim=0):
    input_shape = input.shape
    dim_size = input_shape[dim]
    if dim_size % chunks != 0:
        raise ValueError("Until now, For 'ms_adapter.pytorch.chunk', the value of `input.shape[dim]` "
                       "should be divisible by `chunk`, but got input.shape[{}]:{}, chunks:{}."
                       .format(dim ,dim_size, chunks))

    output = ms.ops.split(input, dim, chunks)
    return output

def mm(input, mat2, *, out=None):
    output_type = input.dtype
    if input.dtype == ms.int32 or input.dtype == ms.int64:
        input = input.astype(ms.float32)
        
    output = ms.ops.matmul(input, mat2)
    output = ms.ops.cast(output, output_type)
    return output


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    _kernel_size = kernel_size + (1,) if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, 1)
    if stride is None:
        _stride = _kernel_size
    else:
        _stride = stride + (1,) if isinstance(stride, tuple) else (stride, stride, 1)
    _padding = padding + (0,) if isinstance(padding, tuple) else (padding, padding, 0)
    _dilation = dilation + (1,) if isinstance(dilation, tuple) else (dilation, dilation, 1)

    dim = input.ndim
    if dim == 3:
        input = input.expand_dims(0)
    input = input.expand_dims(-1)
    out, indices = ms.ops.max_pool3d(input, _kernel_size, _stride, _padding, _dilation, ceil_mode, True)
    if dim == 3:
        out = out.squeeze(0)
        indices = indices.squeeze(0)
    out = out.squeeze(-1)
    indices = indices.squeeze(-1)
    if return_indices:
        return out, indices
    return out


