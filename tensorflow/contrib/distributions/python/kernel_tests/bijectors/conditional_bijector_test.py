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
"""ConditionalBijector Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops.bijectors.conditional_bijector import ConditionalBijector
from tensorflow.contrib.distributions.python.ops.bijectors.chain import Chain
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class _TestBijector(ConditionalBijector):

  def __init__(self):
    super(_TestBijector, self).__init__(
        forward_min_event_ndims=0,
        graph_parents=[],
        is_constant_jacobian=True,
        validate_args=False,
        dtype=dtypes.float32,
        name="test_bijector")

  def _forward(self, _, arg1, arg2):
    raise ValueError("forward", arg1, arg2)

  def _inverse(self, _, arg1, arg2):
    raise ValueError("inverse", arg1, arg2)

  def _inverse_log_det_jacobian(self, _, arg1, arg2):
    raise ValueError("inverse_log_det_jacobian", arg1, arg2)

  def _forward_log_det_jacobian(self, _, arg1, arg2):
    raise ValueError("forward_log_det_jacobian", arg1, arg2)


class _TestPassthroughBijector(_TestBijector):
  def __init__(self, *args, **kwargs):
      super(_TestPassthroughBijector, self).__init__(*args, **kwargs)
      self._called = {
          name: False for name in
          ('forward', 'inverse',
           'inverse_log_det_jacobian', 'forward_log_det_jacobian')
      }

  def _forward(self, _, arg1, arg2):
    self._called['forward'] = True
    return _

  def _inverse(self, _, arg1, arg2):
    self._called['inverse'] = True
    return _

  def _inverse_log_det_jacobian(self, _, arg1, arg2):
    self._called['inverse_log_det_jacobian'] = True
    return _

  def _forward_log_det_jacobian(self, _, arg1, arg2):
    self._called['forward_log_det_jacobian'] = True
    return _


class ConditionalBijectorTest(test.TestCase):

  def testConditionalBijector(self):
    b = _TestBijector()
    for name in ["forward", "inverse"]:
      method = getattr(b, name)
      with self.assertRaisesRegexp(ValueError, name + ".*b1.*b2"):
        method(1., arg1="b1", arg2="b2")

    for name in ["inverse_log_det_jacobian", "forward_log_det_jacobian"]:
      method = getattr(b, name)
      with self.assertRaisesRegexp(ValueError, name + ".*b1.*b2"):
        method(1., event_ndims=0, arg1="b1", arg2="b2")

  def testChainedConditionalBijector(self):
    class ConditionalChain(ConditionalBijector, Chain):
      pass

    test_bijector = _TestBijector()
    passthrough_bijector = _TestPassthroughBijector()
    chain = ConditionalChain((test_bijector, passthrough_bijector))

    name = "forward"
    assert not passthrough_bijector._called[name]
    method = getattr(chain, name)
    with self.assertRaisesRegexp(ValueError, name + ".*b1.*b2"):
      method(
          1.,
          test_bijector={"arg1": "b1", "arg2": "b2"},
          test_passthrough_bijector={"arg1": "b1", "arg2": "b2"})
    assert passthrough_bijector._called[name], name

    name = "forward_log_det_jacobian"
    assert not passthrough_bijector._called[name]
    method = getattr(chain, name)
    with self.assertRaisesRegexp(ValueError, name + ".*b1.*b2"):
      method(
          1.,
          event_ndims=0,
          test_bijector={"arg1": "b1", "arg2": "b2"},
          test_passthrough_bijector={"arg1": "b1", "arg2": "b2"})
    assert passthrough_bijector._called[name], name

    test_bijector = _TestBijector()
    passthrough_bijector = _TestPassthroughBijector()
    chain = ConditionalChain((passthrough_bijector, test_bijector))

    name = "inverse"
    assert not passthrough_bijector._called[name]
    method = getattr(chain, name)
    with self.assertRaisesRegexp(ValueError, name + ".*b1.*b2"):
      method(
          1.,
          test_bijector={"arg1": "b1", "arg2": "b2"},
          test_passthrough_bijector={"arg1": "b1", "arg2": "b2"})
    assert passthrough_bijector._called[name], name

    name = "inverse_log_det_jacobian"
    assert not passthrough_bijector._called[name]
    method = getattr(chain, name)
    with self.assertRaisesRegexp(ValueError, name + ".*b1.*b2"):
      method(
          1.,
          event_ndims=0,
          test_bijector={"arg1": "b1", "arg2": "b2"},
          test_passthrough_bijector={"arg1": "b1", "arg2": "b2"})
    assert passthrough_bijector._called[name], name


if __name__ == "__main__":
  test.main()
