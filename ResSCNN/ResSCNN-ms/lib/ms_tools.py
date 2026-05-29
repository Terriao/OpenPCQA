# -*- coding: utf-8 -*-
import mindspore as ms
import mindspore.ops as ops
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.common.seed import _get_graph_seed
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr

from itertools import repeat
from typing import List, Tuple, Union

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



@constexpr(reuse_result=False)
def _get_seed(op_seed, kernel_name):
    """Get the graph-level seed."""
    return _get_graph_seed(op_seed, kernel_name)

def randn_like(x, seed=None, *, dtype=None):
    r"""
    Returns a new Tensor with given shape and dtype, filled with a sample (or samples) from the standard normal
    distribution.

    Args:
        x (Tensor): Input Tensor to specify the output shape and its default dtype.
        seed (int, optional): Random seed, must be greater or equal to 0. Default: None, and 0 will be used.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): Designated tensor dtype, it must be float type. If None,
            `mindspore.float32` will be used. Default: None.

    Returns:
        Tensor, with the designated shape and dtype, filled with a sample (or samples) from the
        "standard normal" distribution.

    Raises:
        TypeError: `seed` is not a non-negative integer.
        ValueError: If `dtype` is not a `mstype.float_type`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops
        >>> a = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> print(ops.randn_like(x, dtype=ms.float32))
        [[ 0.30639967 -0.42438635 -0.20454668]
         [-0.4287376   1.3054721   0.64747655]]
    """
    if dtype is None:
        dtype = x.dtype
    
    shape = x.shape
    cast_ = P.Cast()
    seed1, seed2 = _get_seed(seed, 'randn_like')
    rand_op = P.StandardNormal(seed1, seed2)
    output = rand_op(shape)
    return cast_(output, dtype)



def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape

    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h

def sparse_quantize(coords,
                    voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *,
                    return_index: bool = False,
                    return_inverse: bool = False) -> List[np.ndarray]:
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(ravel_hash(coords),
                                            return_index=True,
                                            return_inverse=True)
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs

