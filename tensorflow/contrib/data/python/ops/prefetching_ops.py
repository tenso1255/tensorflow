# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Python wrapper for prefetching_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from tensorflow.contrib.data.python.ops import contrib_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.data.python.ops import gen_dataset_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.eager import context
from tensorflow.python.framework import device as framework_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops as core_gen_dataset_ops
from tensorflow.python.ops import resource_variable_ops


def function_buffering_resource(string_arg,
                                target_device,
                                f,
                                buffer_size,
                                output_types,
                                container="",
                                shared_name=None,
                                name=None):
  """Creates a FunctionBufferingResource.

  A FunctionBufferingResource fills up a buffer by calling a function `f` on
  `target_device`. `f` should take in only a single string argument as input.

  Args:
    string_arg: The single string argument to the function.
    target_device: The device to run `f` on.
    f: The function to be executed.
    buffer_size: Size of the buffer to be populated.
    output_types: The output types generated by the function.
    container: (Optional) string. Defaults to "".
    shared_name: (Optional) string.
    name: (Optional) string to name the op.

  Returns:
    Handle to a FunctionBufferingResource.
  """
  if shared_name is None:
    shared_name = ""
  return gen_dataset_ops.function_buffering_resource(
      string_arg=string_arg,
      target_device=target_device,
      shared_name=shared_name,
      f=f,
      buffer_size=buffer_size,
      container=container,
      name=name,
      output_types=output_types)


def function_buffering_resource_get_next(function_buffer_resource,
                                         output_types,
                                         name=None):
  return gen_dataset_ops.function_buffering_resource_get_next(
      function_buffer_resource=function_buffer_resource,
      output_types=output_types,
      name=name)


def function_buffering_resource_reset(function_buffer_resource, name=None):
  return gen_dataset_ops.function_buffering_resource_reset(
      function_buffer_resource=function_buffer_resource, name=name)


# pylint: disable=protected-access
class _PrefetchToDeviceIterator(object):
  """A replacement for @{tf.data.Iterator} that prefetches to another device.

  Args:
    input_dataset: The input dataset
    one_shot: If true, we make a one shot iterator that's already initialized.
    device: A fully specified device string where we want to prefetch to
    buffer_size: Size of the prefetching buffer.
    shared_name: (Optional.) If non-empty, the returned iterator will be
        shared under the given name across multiple sessions that share the
        same devices (e.g. when using a remote server).

  Returns:
    An Iterator type object.
  """

  def __init__(self,
               input_dataset,
               one_shot,
               device,
               buffer_size,
               shared_name=None):
    self._input_dataset = input_dataset
    self._get_next_call_count = 0
    self._one_shot = one_shot
    if shared_name is None:
      shared_name = ""

    if self._one_shot:
      self._input_iterator = input_dataset.make_one_shot_iterator()
    else:
      self._input_iterator = iterator_ops.Iterator.from_structure(
          self._input_dataset.output_types, self._input_dataset.output_shapes,
          shared_name, self._input_dataset.output_classes)
    input_iterator_handle = self._input_iterator.string_handle()

    @function.Defun(dtypes.string)
    def _prefetch_fn(handle):
      """Prefetches one element from `input_iterator`."""
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          handle, self._input_iterator.output_types,
          self._input_iterator.output_shapes,
          self._input_iterator.output_classes)
      ret = remote_iterator.get_next()
      return nest.flatten(sparse.serialize_sparse_tensors(ret))

    iterator_device = gen_dataset_ops.iterator_get_device(
        self._input_iterator._iterator_resource)

    with ops.device(device):
      self._buffering_resource = function_buffering_resource(
          f=_prefetch_fn,
          target_device=iterator_device,
          string_arg=input_iterator_handle,
          buffer_size=buffer_size,
          shared_name=shared_name,
          output_types=nest.flatten(
              sparse.as_dense_types(self._input_dataset.output_types,
                                    self._input_dataset.output_classes)))

    if not self._one_shot:
      reset_op = function_buffering_resource_reset(self._buffering_resource)
      with ops.control_dependencies([reset_op]):
        self._initializer = self._input_iterator.make_initializer(
            self._input_dataset)

  def get_next(self, name=None):
    """See @{tf.data.Iterator.get_next}."""
    self._get_next_call_count += 1
    if self._get_next_call_count > iterator_ops.GET_NEXT_CALL_WARNING_THRESHOLD:
      warnings.warn(iterator_ops.GET_NEXT_CALL_WARNING_MESSAGE)

    flat_ret = gen_dataset_ops.function_buffering_resource_get_next(
        self._buffering_resource,
        output_types=nest.flatten(sparse.as_dense_types(
            self.output_types, self.output_classes)), name=name)

    ret = sparse.deserialize_sparse_tensors(
        nest.pack_sequence_as(self.output_types, flat_ret),
        self.output_types, self.output_shapes, self.output_classes)

    for tensor, shape in zip(
        nest.flatten(ret), nest.flatten(self.output_shapes)):
      if isinstance(tensor, ops.Tensor):
        tensor.set_shape(shape)

    return ret

  @property
  def initializer(self):
    if self._one_shot:
      raise NotImplementedError("Can't initialize a one_shot_iterator")
    return self._initializer

  @property
  def output_classes(self):
    return self._input_dataset.output_classes

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_types(self):
    return self._input_dataset.output_types


class _PrefetchToDeviceEagerIterator(iterator_ops.EagerIterator):
  """A replacement for @{tf.data.Iterator} that prefetches to another device.

  Args:
    input_dataset: The input dataset
    one_shot: If true, we make a one shot iterator that's already initialized.
    device: A fully specified device string where we want to prefetch to
    buffer_size: Size of the prefetching buffer.
    shared_name: (Optional.) If non-empty, the returned iterator will be
        shared under the given name across multiple sessions that share the
        same devices (e.g. when using a remote server).

  Returns:
    An Iterator type object.
  """

  def __init__(self,
               input_dataset,
               device,
               buffer_size):
    with ops.device("/device:CPU:0"):
      super(_PrefetchToDeviceEagerIterator, self).__init__(input_dataset)
      input_iterator_handle = core_gen_dataset_ops.iterator_to_string_handle(
          self._resource)

    self._device = device

    @function.Defun(dtypes.string)
    def _prefetch_fn(handle):
      """Prefetches one element from `input_iterator`."""
      remote_iterator = iterator_ops.Iterator.from_string_handle(
          handle, self.output_types, self.output_shapes, self.output_classes)
      ret = remote_iterator.get_next()
      return nest.flatten(sparse.serialize_sparse_tensors(ret))

    _prefetch_fn.add_to_graph(None)

    with ops.device(device):
      self._buffering_resource = function_buffering_resource(
          f=_prefetch_fn,
          output_types=self._flat_output_types,
          target_device=gen_dataset_ops.iterator_get_device(self._resource),
          string_arg=input_iterator_handle,
          buffer_size=buffer_size,
          shared_name=iterator_ops._generate_shared_name(
              "function_buffer_resource"))

  def _next_internal(self):
    """Returns a nested structure of `tf.Tensor`s containing the next element.
    """
    # This runs in sync mode as iterators use an error status to communicate
    # that there is no more data to iterate over.
    # TODO(b/77291417): Fix
    with context.execution_mode(context.SYNC):
      with ops.device(self._device):
        ret = gen_dataset_ops.function_buffering_resource_get_next(
            function_buffer_resource=self._buffering_resource,
            output_types=self._flat_output_types)
      return sparse.deserialize_sparse_tensors(
          nest.pack_sequence_as(self._output_types, ret), self._output_types,
          self._output_shapes, self._output_classes)
# pylint: enable=protected-access


class _PrefetchToDeviceDataset(dataset_ops.Dataset):
  """A `Dataset` whose iterator prefetches elements to another device."""

  def __init__(self, input_dataset, device, buffer_size):
    self._input_dataset = input_dataset
    self._device = device
    self._buffer_size = buffer_size if buffer_size is not None else 1

  # The static analysis cannot tell that the eager iterator's superclass has
  # a `next()` method.
  # pylint: disable=non-iterator-returned
  def __iter__(self):
    """Creates an `Iterator` for enumerating the elements of this dataset.

    The returned iterator implements the Python iterator protocol and therefore
    can only be used in eager mode.

    Returns:
      An `Iterator` over the elements of this dataset.

    Raises:
      RuntimeError: If eager execution is enabled.
    """
    if context.executing_eagerly():
      return _PrefetchToDeviceEagerIterator(self._input_dataset, self._device,
                                            self._buffer_size)
    else:
      raise RuntimeError("dataset.__iter__() is only supported when eager "
                         "execution is enabled.")
  # pylint: enable=non-iterator-returned

  def make_one_shot_iterator(self):
    if context.executing_eagerly():
      return _PrefetchToDeviceEagerIterator(self._input_dataset, self._device,
                                            self._buffer_size)
    else:
      return _PrefetchToDeviceIterator(self._input_dataset, one_shot=True,
                                       device=self._device,
                                       buffer_size=self._buffer_size)

  def make_initializable_iterator(self, shared_name=None):
    return _PrefetchToDeviceIterator(
        self._input_dataset,
        one_shot=False,
        device=self._device,
        buffer_size=self._buffer_size,
        shared_name=shared_name)

  def _as_variant_tensor(self):
    # TODO(mrry): Raise this error earlier (e.g. when one of the Dataset
    # transformation methods is called.
    # TODO(mrry): Investigate support for chaining further transformations after
    # the prefetch, including GPU support.
    raise NotImplementedError("`prefetch_to_device()` must be the last "
                              "transformation in a dataset pipeline.")

  @property
  def output_types(self):
    return self._input_dataset.output_types

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_classes(self):
    return self._input_dataset.output_classes


def prefetch_to_device(device, buffer_size=None):
  """A transformation that prefetches dataset values to the given `device`.

  NOTE: Although the transformation creates a @{tf.data.Dataset}, the
  transformation must be the final `Dataset` in the input pipeline.

  Args:
    device: A string. The name of a device to which elements will be prefetched.
    buffer_size: (Optional.) The number of elements to buffer on `device`.
      Defaults to an automatically chosen value.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.
  """
  def _apply_fn(dataset):
    return _PrefetchToDeviceDataset(dataset, device, buffer_size)

  return _apply_fn


def copy_to_device(target_device, source_device="/cpu:0"):
  """A transformation that copies dataset elements to the given `target_device`.

  Args:
    target_device: The name of a device to which elements will be copied.
    source_device: The original device on which `input_dataset` will be placed.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.
  """

  def _apply_fn(dataset):
    return _CopyToDeviceDataset(
        dataset, target_device=target_device, source_device=source_device)

  return _apply_fn


# TODO(rohanj): Use the _input_hostmem attr on the RemoteCall ops to indicate
# all inputs to the Op are in host memory, thereby avoiding some unecessary
# Sends and Recvs.
class _CopyToDeviceDataset(dataset_ops.Dataset):
  """A `Dataset` that copies elements to another device."""

  def __init__(self, input_dataset, target_device, source_device="/cpu:0"):
    """Constructs a _CopyToDeviceDataset.

    Args:
      input_dataset: `Dataset` to be copied
      target_device: The name of the device to which elements would be copied.
      source_device: Device where input_dataset would be placed.
    """
    self._input_dataset = input_dataset
    self._target_device = target_device
    spec = framework_device.DeviceSpec().from_string(self._target_device)
    self._is_gpu_target = (spec.device_type == "GPU")
    self._source_device_string = source_device
    self._source_device = ops.convert_to_tensor(source_device)

    self._flat_output_shapes = nest.flatten(
        sparse.as_dense_shapes(self._input_dataset.output_shapes,
                               self._input_dataset.output_classes))
    self._flat_output_types = nest.flatten(
        sparse.as_dense_types(self._input_dataset.output_types,
                              self._input_dataset.output_classes))

    @function.Defun()
    def _init_func():
      """Creates an iterator for the input dataset.

      Returns:
        A `string` tensor that encapsulates the iterator created.
      """
      # pylint: disable=protected-access
      ds_variant = self._input_dataset._as_variant_tensor()
      resource = core_gen_dataset_ops.anonymous_iterator(
          output_types=self._flat_output_types,
          output_shapes=self._flat_output_shapes)
      with ops.control_dependencies(
          [core_gen_dataset_ops.make_iterator(ds_variant, resource)]):
        return core_gen_dataset_ops.iterator_to_string_handle(resource)

    @function.Defun()
    def _remote_init_func():
      return functional_ops.remote_call(
          target=self._source_device,
          args=_init_func.captured_inputs,
          Tout=[dtypes.string],
          f=_init_func)

    self._init_func = _remote_init_func
    self._init_captured_args = _remote_init_func.captured_inputs

    @function.Defun(dtypes.string)
    def _next_func(string_handle):
      """Calls get_next for created iterator.

      Args:
        string_handle: An iterator string handle created by _init_func
      Returns:
        The elements generated from `input_dataset`
      """
      with ops.device(self._source_device_string):
        iterator = iterator_ops.Iterator.from_string_handle(
            string_handle, self.output_types, self.output_shapes,
            self.output_classes)
      ret = iterator.get_next()
      return nest.flatten(sparse.serialize_sparse_tensors(ret))

    @function.Defun(dtypes.string)
    def _remote_next_func(string_handle):
      return functional_ops.remote_call(
          target=self._source_device,
          args=[string_handle] + _next_func.captured_inputs,
          Tout=self._flat_output_types,
          f=_next_func)

    self._next_func = _remote_next_func
    self._next_captured_args = _remote_next_func.captured_inputs

    @function.Defun(dtypes.string)
    def _finalize_func(string_handle):
      """Destroys the iterator resource created.

      Args:
        string_handle: An iterator string handle created by _init_func
      Returns:
        Tensor constant 0
      """
      iterator_resource = core_gen_dataset_ops.iterator_from_string_handle_v2(
          string_handle,
          output_types=self._flat_output_types,
          output_shapes=self._flat_output_shapes)
      with ops.control_dependencies([
          resource_variable_ops.destroy_resource_op(
              iterator_resource, ignore_lookup_error=True)]):
        return array_ops.constant(0, dtypes.int64)

    @function.Defun(dtypes.string)
    def _remote_finalize_func(string_handle):
      return functional_ops.remote_call(
          target=self._source_device,
          args=[string_handle] + _finalize_func.captured_inputs,
          Tout=[dtypes.int64],
          f=_finalize_func)

    self._finalize_func = _remote_finalize_func
    self._finalize_captured_args = _remote_finalize_func.captured_inputs
    # pylint: enable=protected-scope

  # The one_shot_iterator implementation needs a 0 arg _make_dataset function
  # that thereby captures all the inputs required to create the dataset. Since
  # there are strings that are inputs to the GeneratorDataset which can't be
  # placed on a GPU, this fails for the GPU case. Therefore, disabling it for
  # GPU
  def make_one_shot_iterator(self):
    if self._is_gpu_target:
      raise ValueError("Cannot create a one shot iterator when using "
                       "`tf.contrib.data.copy_to_device()` on GPU. Please use "
                       "`Dataset.make_initializable_iterator()` instead.")
    else:
      return super(_CopyToDeviceDataset, self).make_one_shot_iterator()

  def _as_variant_tensor(self):
    with ops.device(self._target_device):
      return core_gen_dataset_ops.generator_dataset(
          self._init_captured_args,
          self._next_captured_args,
          self._finalize_captured_args,
          init_func=self._init_func,
          next_func=self._next_func,
          finalize_func=self._finalize_func,
          output_types=self._flat_output_types,
          output_shapes=self._flat_output_shapes)

  @property
  def output_types(self):
    return self._input_dataset.output_types

  @property
  def output_shapes(self):
    return self._input_dataset.output_shapes

  @property
  def output_classes(self):
    return self._input_dataset.output_classes
