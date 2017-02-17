# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""ConditionalBijector base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops.bijectors import bijector


__all__ = ["ConditionalBijector"]


class ConditionalBijector(bijector.Bijector):
  """Conditional Bijector is a Bijector that allows intrinsic conditioning."""

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def forward(self, x, name="forward", **condition_kwargs):
    return self._call_forward(x, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def inverse(self, y, name="inverse", **condition_kwargs):
    return self._call_inverse(y, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def inverse_log_det_jacobian(
      self, y, name="inverse_log_det_jacobian", **condition_kwargs):
    return self._call_inverse_log_det_jacobian(y, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def inverse_and_inverse_log_det_jacobian(
      self, y, name="inverse_and_inverse_log_det_jacobian", **condition_kwargs):
    return self._call_inverse_and_inverse_log_det_jacobian(
        y, name, **condition_kwargs)

  @distribution_util.AppendDocstring(kwargs_dict={
      "**condition_kwargs":
      "Named arguments forwarded to subclass implementation."})
  def forward_log_det_jacobian(
      self, x, name="forward_log_det_jacobian", **condition_kwargs):
    return self._call_forward_log_det_jacobian(x, name, **condition_kwargs)
