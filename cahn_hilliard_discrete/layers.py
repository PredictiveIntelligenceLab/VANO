from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

from flax.linen.initializers import lecun_normal
from flax.linen.initializers import zeros
from flax.linen.module import compact
from flax.linen.module import Module
# from flax.linen.dtypes import promote_dtype
from jax import vmap, eval_shape
from jax import lax
from jax import ShapedArray
import jax.numpy as jnp
import numpy as np


from jax import random
from jax.nn.initializers import glorot_normal, normal, zeros

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]

default_kernel_init = lecun_normal()

PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]


def canonicalize_dtype(*args,
                       dtype: Optional[Dtype] = None,
                       inexact: bool = True) -> Dtype:
  """Canonicalize an optional dtype to the definitive dtype.
  If the ``dtype`` is None this function will infer the dtype. If it is not
  None it will be returned unmodified or an exceptions is raised if the dtype
  is invalid.
  from the input arguments using ``jnp.result_type``.
  Args:
    *args: JAX array compatible values. None values
      are ignored.
    dtype: Optional dtype override. If specified the arguments are cast to
      the specified dtype instead and dtype inference is disabled.
    inexact: When True, the output dtype must be a subdtype
    of `jnp.inexact`. Inexact dtypes are real or complex floating points. This
    is useful when you want to apply operations that don't work directly on
    integers like taking a mean for example.
  Returns:
    The dtype that *args should be cast to.
  """
  if dtype is None:
    args_filtered = [jnp.asarray(x) for x in args if x is not None]
    dtype = jnp.result_type(*args_filtered)
    if inexact and not jnp.issubdtype(dtype, jnp.inexact):
      dtype = jnp.promote_types(jnp.float32, dtype)
  if inexact and not jnp.issubdtype(dtype, jnp.inexact):
    raise ValueError(f'Dtype must be inexact: {dtype}')
  return dtype


def promote_dtype(*args, dtype=None, inexact=True) -> List[Array]:
  """"Promotes input arguments to a specified or inferred dtype.
  All args are cast to the same dtype. See ``canonicalize_dtype`` for how
  this dtype is determined.
  The behavior of promote_dtype is mostly a convinience wrapper around
  ``jax.numpy.promote_types``. The differences being that it automatically casts
  all input to the inferred dtypes, allows inference to be overridden by a
  forced dtype, and has an optional check to garantuee the resulting dtype is
  inexact.
  Args:
    *args: JAX array compatible values. None values
      are returned as is.
    dtype: Optional dtype override. If specified the arguments are cast to
      the specified dtype instead and dtype inference is disabled.
    inexact: When True, the output dtype must be a subdtype
    of `jnp.inexact`. Inexact dtypes are real or complex floating points. This
    is useful when you want to apply operations that don't work directly on
    integers like taking a mean for example.
  Returns:
    The arguments cast to arrays of the same dtype.
  """
  dtype = canonicalize_dtype(*args, dtype=dtype, inexact=inexact)
  return [jnp.asarray(x, dtype) if x is not None else None
          for x in args]


def _normalize_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


def _canonicalize_tuple(x: Union[Sequence[int], int]) -> Tuple[int, ...]:
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


def _weight_fact(init_fn):
    
    def init(key, shape, mean=1.0, stddev=0.01):
        key1, key2 = random.split(key)
        w = init_fn(key1, shape)
        g = mean + normal(stddev)(key2, (shape[-1],))
        g = jnp.exp(g)
        v = w / g
        return g, v

    return init



class Dense(Module):
    features: int

    @compact
    def __call__(self, x):
        g, v = self.param('kernel',
                          _weight_fact(glorot_normal()),
                          (x.shape[-1], self.features))
        kernel = g * v
        bias = self.param('bias', zeros, (self.features,))
        y = jnp.dot(x, kernel) + bias
        return y


def _normalize_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


def _canonicalize_tuple(x: Union[Sequence[int], int]) -> Tuple[int, ...]:
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)




def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
  """"Canonicalizes conv padding to a jax.lax supported format."""
  if isinstance(padding, str):
    return padding
  if isinstance(padding, int):
    return [(padding, padding)] * rank
  if isinstance(padding, Sequence) and len(padding) == rank:
    new_pad = []
    for p in padding:
      if isinstance(p, int):
        new_pad.append((p, p))
      elif isinstance(p, tuple) and len(p) == 2:
        new_pad.append(p)
      else:
        break
    if len(new_pad) == rank:
      return new_pad
  raise ValueError(
    f'Invalid padding format: {padding}, should be str, int,'
    f' or a sequence of len {rank} where each element is an'
    f' int or pair of ints.')

class _Conv(Module):
    
  features: int
  kernel_size: Sequence[int]
  strides: Union[None, int, Sequence[int]] = 1
  padding: PaddingLike = 'SAME'
  input_dilation: Union[None, int, Sequence[int]] = 1
  kernel_dilation: Union[None, int, Sequence[int]] = 1
  feature_group_count: int = 1
  use_bias: bool = True
  mask: Optional[Array] = None
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  weight_fact: bool = False

  @property
  def shared_weights(self) -> bool:
    """Defines whether weights are shared or not between different pixels.

    Returns:
      `True` to use shared weights in convolution (regular convolution).
      `False` to use different weights at different pixels, a.k.a.
      "locally connected layer", "unshared convolution", or "local convolution".

    """
    ...

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a (potentially unshared) convolution to the inputs.

    Args:
      inputs: input data with dimensions (*batch_dims, spatial_dims...,
        features). This is the channels-last convention, i.e. NHWC for a 2d
        convolution and NDHWC for a 3D convolution. Note: this is different from
        the input convention used by `lax.conv_general_dilated`, which puts the
        spatial dimensions last.
        Note: If the input has more than 1 batch dimension, all batch dimensions
        are flattened into a single dimension for the convolution and restored
        before returning.  In some cases directly vmap'ing the layer may yield
        better performance than this default flattening approach.  If the input
        lacks a batch dimension it will be added for the convolution and removed
        n return, an allowance made to enable writing single-example code.

    Returns:
      The convolved data.
    """

    if isinstance(self.kernel_size, int):
      raise TypeError('Expected Conv kernel_size to be a'
                      ' tuple/list of integers (eg.: [3, 3]) but got'
                      f' {self.kernel_size}.')
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(x: Optional[Union[int, Sequence[int]]]) -> (
        Tuple[int, ...]):
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return tuple(x)

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (
          (total_batch_size,) + inputs.shape[num_batch_dimensions:])
      inputs = jnp.reshape(inputs, flat_input_shape)

    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax == 'CIRCULAR':
      kernel_size_dilated = [
          (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
      ]
      zero_pad: List[Tuple[int, int]] = [(0, 0)]
      pads = (zero_pad + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] +
              [(0, 0)])
      inputs = jnp.pad(inputs, pads, mode='wrap')
      padding_lax = 'VALID'
    elif padding_lax == 'CAUSAL':
      if len(kernel_size) != 1:
        raise ValueError(
            'Causal padding is only implemented for 1D convolutions.')
      left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
      pads = [(0, 0), (left_pad, 0), (0, 0)]
      inputs = jnp.pad(inputs, pads)
      padding_lax = 'VALID'

    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    in_features = jnp.shape(inputs)[-1]

    if self.shared_weights:
      # One shared convolutional kernel for all pixels in the output.
      assert in_features % self.feature_group_count == 0
      kernel_shape = kernel_size + (
          in_features // self.feature_group_count, self.features)

    else:
      if self.feature_group_count != 1:
        raise NotImplementedError(
            f'`lax.conv_general_dilated_local` does not support '
            f'`feature_group_count != 1`, got `{self.feature_group_count}`.'
        )

      # Need to know the spatial output shape of a standard convolution to
      # create the unshared convolution kernel.
      conv_output_shape = eval_shape(
          lambda lhs, rhs: lax.conv_general_dilated(  # pylint: disable=g-long-lambda
              lhs=lhs,
              rhs=rhs,
              window_strides=strides,
              padding=padding_lax,
              dimension_numbers=dimension_numbers,
              lhs_dilation=input_dilation,
              rhs_dilation=kernel_dilation,
          ),
          inputs,
          ShapedArray(kernel_size + (in_features, self.features), inputs.dtype)
      ).shape

      # One (unshared) convolutional kernel per each pixel in the output.
      kernel_shape = conv_output_shape[1:-1] + (np.prod(kernel_size) *
                                                in_features, self.features)

    if self.mask is not None and self.mask.shape != kernel_shape:
      raise ValueError('Mask needs to have the same shape as weights. '
                       f'Shapes are: {self.mask.shape}, {kernel_shape}')
    
    if self.weight_fact:
      g, v = self.param('kernel', _weight_fact(self.kernel_init), kernel_shape)
            
      kernel = g * v
    
    else: 
      kernel = self.param('kernel', self.kernel_init, kernel_shape,
                        self.param_dtype)
        
    if self.mask is not None:
      kernel *= self.mask

    if self.use_bias:
      if self.shared_weights:
        # One bias weight per output channel, shared between pixels.
        bias_shape = (self.features,)
      else:
        # One bias weight per output entry, unshared betwen pixels.
        bias_shape = conv_output_shape[1:]

      bias = self.param('bias', self.bias_init, bias_shape, self.param_dtype)
    else:
      bias = None

    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
    if self.shared_weights:
      y = lax.conv_general_dilated(
          inputs,
          kernel,
          strides,
          padding_lax,
          lhs_dilation=input_dilation,
          rhs_dilation=kernel_dilation,
          dimension_numbers=dimension_numbers,
          feature_group_count=self.feature_group_count,
          precision=self.precision
      )
    else:
      y = lax.conv_general_dilated_local(
          lhs=inputs,
          rhs=kernel,
          window_strides=strides,
          padding=padding_lax,
          filter_shape=kernel_size,
          lhs_dilation=input_dilation,
          rhs_dilation=kernel_dilation,
          dimension_numbers=dimension_numbers,
          precision=self.precision
      )

    if self.use_bias:
      bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
      y += bias

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)
    return y


class Conv(_Conv):

  @property
  def shared_weights(self) -> bool:
    return True

class ConvTranspose(Module):
  """Convolution Module wrapping lax.conv_transpose.
  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer. For all other cases, it must
      be a sequence of integers.
    strides: a sequence of `n` integers, representing the inter-window strides.
    padding: either the string `'SAME'`, the string `'VALID'`, the string
      `'CIRCULAR'` (periodic boundary conditions), or a sequence of `n` `(low,
      high)` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpeted as applying the same padding
      in all dims and passign a single int in a sequence causes the same padding
      to be used on both sides.
    kernel_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel. Convolution with kernel dilation is also known as 'atrous
      convolution'.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
  """
  features: int
  kernel_size: Union[int, Tuple[int, ...]]
  strides: Optional[Tuple[int, ...]] = None
  padding: PaddingLike = 'SAME'
  kernel_dilation: Optional[Sequence[int]] = None
  use_bias: bool = True
  mask: Optional[Array] = None
  dtype: Dtype = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  weight_fact: bool = False

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a transposed convolution to the inputs.
    Behaviour mirrors of `jax.lax.conv_transpose`.
    Args:
      inputs: input data with dimensions (*batch_dims, spatial_dims...,
        features). This is the channels-last convention, i.e. NHWC for a 2d
        convolution and NDHWC for a 3D convolution. Note: this is different from
        the input convention used by `lax.conv_general_dilated`, which puts the
        spatial dimensions last.
        Note: If the input has more than 1 batch dimension, all batch dimensions
        are flattened into a single dimension for the convolution and restored
        before returning.  In some cases directly vmap'ing the layer may yield
        better performance than this default flattening approach.  If the input
        lacks a batch dimension it will be added for the convolution and removed
        n return, an allowance made to enable writing single-example code.
    Returns:
      The convolved data.
    """
    kernel_size: Tuple[int, ...]
    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = self.kernel_size

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (
          (total_batch_size,) + inputs.shape[num_batch_dimensions:])
      inputs = jnp.reshape(inputs, flat_input_shape)

    strides: Tuple[int, ...]
    strides = self.strides or (1,) * (inputs.ndim - 2)

    in_features = jnp.shape(inputs)[-1]
    kernel_shape = kernel_size + (in_features, self.features)

    if self.mask is not None and self.mask.shape != kernel_shape:
      raise ValueError('Mask needs to have the same shape as weights. '
                       f'Shapes are: {self.mask.shape}, {kernel_shape}')

    if self.weight_fact:
      g, v = self.param('kernel', _weight_fact(self.kernel_init), kernel_shape)
            
      kernel = g * v
    
    else: 
      kernel = self.param('kernel', self.kernel_init, kernel_shape,
                        self.param_dtype)

    if self.mask is not None:
      kernel *= self.mask

    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax == 'CIRCULAR':
      padding_lax = 'VALID'

    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,),
                        self.param_dtype)
    else:
      bias = None

    inputs, kernel, bias = promote_dtype(inputs, kernel, bias,
                                         dtype=self.dtype)

    y = lax.conv_transpose(
        inputs,
        kernel,
        strides,
        padding_lax,
        rhs_dilation=self.kernel_dilation,
        precision=self.precision)

    if self.padding == 'CIRCULAR':
      # For circular padding, we need to identify the size of the final output
      # ("period") along each spatial dimension, pad each dimension to an
      # integer number of periods, and wrap the array periodically around each
      # dimension. Padding should be done in such a way that the start of the
      # original input data inside the padded array is located at integer
      # number of periods - otherwise the result would be circularly shifted.

      # Compute period along each spatial dimension - it's input size scaled
      # by the stride.
      scaled_x_dims = [
          x_dim * stride for x_dim, stride in zip(jnp.shape(inputs)[1:-1], strides)
      ]
      # Compute difference between the current size of y and the final output
      # size, and complement this difference to 2 * period - that gives how
      # much we need to pad.
      size_diffs = [
          -(y_dim - x_dim) % (2 * x_dim)
          for y_dim, x_dim in zip(y.shape[1:-1], scaled_x_dims)
      ]
      # Divide the padding equaly between left and right. The choice to put
      # "+1" on the left (and not on the right) represents a convention for
      # aligning even-sized kernels.
      total_pad = [
          ((size_diff + 1) // 2, size_diff // 2) for size_diff in size_diffs
      ]
      y = jnp.pad(y, [(0, 0)] + total_pad + [(0, 0)])
      # Wrap the result periodically around each spatial dimension,
      # one by one.
      for i in range(1, y.ndim - 1):
        y = y.reshape(y.shape[:i] + (-1, scaled_x_dims[i - 1]) +
                      y.shape[i + 1:])
        y = y.sum(axis=i)

    if self.use_bias:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)

    return y

# Helper functions
BOX_OFFSETS = [jnp.array([[i for i in [0, 1]]]).T,
               jnp.array([[i,j] for i in [0, 1] for j in [0, 1]]),
               jnp.array([[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]),
               jnp.array([[i,j,k,l] for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1]]),
               jnp.array([[i,j,k,l,m] for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1] for m in [0, 1]])]

def get_voxel_vertices(x, n):
    dim = x.shape[0]
    grid_size = 1.0/n
    xl = jnp.int32(jnp.floor(x*n))
    xu = jnp.int32(jnp.ceil(x*n))
    voxel_min_vertex = xl*grid_size
    voxel_max_vertex = voxel_min_vertex + jnp.ones(dim)*grid_size
    voxel_indices = xl + BOX_OFFSETS[dim-1]
    return voxel_indices, voxel_min_vertex, voxel_max_vertex

def encode_vertex(v, T):
    dim = v.shape[0]
    primes = jnp.int32(jnp.array([1, 19349663, 915850651, 2147483647]))[:dim]
    def hash_fn(carry, i):
        carry = jnp.bitwise_xor(v[i]*primes[i], carry)
        return carry, None
    res, _ = lax.scan(hash_fn, 0, jnp.arange(dim))
    return jnp.mod(res,T)

def get_vertex_features(w, rows, i):
    return w[i,rows,:]
    
def bilinear_interp(x, vertex_features, voxel_min_vertex, voxel_max_vertex):
    # source: https://en.wikipedia.org/wiki/Bilinear_interpolation
    w1 = (voxel_max_vertex - x)/(voxel_max_vertex-voxel_min_vertex)
    w2 = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex)
    f1 = w1[0]*vertex_features[0,:] + w2[0]*vertex_features[2,:]
    f2 = w1[0]*vertex_features[1,:] + w2[0]*vertex_features[3,:]
    interp_val = w1[1]*f1 + w2[1]*f2
    return interp_val


class FourierEnc(Module):
    embed_scale: float = 1.0
    embed_dim: int = 128

    @compact
    def __call__(self, x):
        kernel = self.param('kernel', normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2))
        y = jnp.concatenate([jnp.cos(jnp.dot(x, kernel)),
                             jnp.sin(jnp.dot(x, kernel))], axis=-1)
        return y

def uniform_init(minval=-0.0001, maxval=0.0001):
    def init(key, shape):
        w = random.uniform(key, shape, minval=minval, maxval=maxval)
        return w
    return init

class MultiresEnc(Module):
    num_levels: int=16
    min_res: int=16
    max_res: int=1024
    num_hash_slots: int=2**16
    num_hash_features: int=2
        
    @compact
    def __call__(self, x):
        kernel = self.param('kernel', uniform_init(), (self.num_levels,
                                                       self.num_hash_slots,
                                                       self.num_hash_features))
        
        b = jnp.exp((jnp.log(self.max_res)-jnp.log(self.min_res))/(self.num_levels-1))
        N = jnp.int32(jnp.floor(self.min_res*b**jnp.arange(1,self.num_levels+1)))
        
        # Get nearby vertices
        voxel_indices, \
        voxel_min_vertex, \
        voxel_max_vertex = vmap(get_voxel_vertices, in_axes=(None,0))(x, N)
        # Compute hash table encodings
        vertex_encodings = vmap(vmap(encode_vertex, in_axes=(0,None)), in_axes=(0,None))(voxel_indices, 
                                                                                         self.num_hash_slots)
        # Look-up vertex features
        vertex_features = vmap(get_vertex_features, in_axes=(None,0,0))(kernel, 
                                                                        vertex_encodings, 
                                                                        jnp.arange(self.num_levels))
        # Get input features via interpolation
        x = vmap(bilinear_interp, in_axes=(None,0,0,0))(x,
                                                        vertex_features, 
                                                        voxel_min_vertex, 
                                                        voxel_max_vertex) 
        return x.flatten()
