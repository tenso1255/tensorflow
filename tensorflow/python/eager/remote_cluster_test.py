# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for remote eager execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import server_lib

JOB_NAME = "remote_device"


def get_server_def(job_name, local_server_port, remote_server_addresses,
                   task_index):
  """Returns a server def with a single job + multiple tasks."""
  cluster_def = cluster_pb2.ClusterDef()
  job_def = cluster_def.job.add()
  job_def.name = job_name
  job_def.tasks[0] = "localhost:%d" % local_server_port

  for i, remote_server_address in enumerate(remote_server_addresses, start=1):
    job_def.tasks[i] = remote_server_address

  server_def = tensorflow_server_pb2.ServerDef(
      cluster=cluster_def,
      job_name=job_name,
      task_index=task_index,
      protocol="grpc")

  return server_def


class DynamicClusterTest(test.TestCase, parameterized.TestCase):

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(DynamicClusterTest, self).__init__(methodName)
    self._cached_server1 = server_lib.Server.create_local_server()
    self._cached_server2 = server_lib.Server.create_local_server()
    self._cached_server3 = server_lib.Server.create_local_server()
    self._cached_server4 = server_lib.Server.create_local_server()

    self._cached_server1_target = self._cached_server1.target[len("grpc://"):]
    self._cached_server2_target = self._cached_server2.target[len("grpc://"):]
    self._cached_server3_target = self._cached_server3.target[len("grpc://"):]
    self._cached_server4_target = self._cached_server4.target[len("grpc://"):]

    self.server_def_s1 = get_server_def(
        JOB_NAME,
        local_server_port=0,
        remote_server_addresses=[self._cached_server1_target],
        task_index=0)
    self.server_def_s1_s2 = get_server_def(
        JOB_NAME,
        local_server_port=0,
        remote_server_addresses=[
            self._cached_server1_target, self._cached_server2_target
        ],
        task_index=0)
    self.server_def_s1_s3 = get_server_def(
        JOB_NAME,
        local_server_port=0,
        remote_server_addresses=[
            self._cached_server1_target, self._cached_server3_target
        ],
        task_index=0)
    self.server_def_s4_s3 = get_server_def(
        JOB_NAME,
        local_server_port=0,
        remote_server_addresses=[
            self._cached_server4_target, self._cached_server3_target
        ],
        task_index=0)
    self.server_def_s1_s2_s3 = get_server_def(
        JOB_NAME,
        local_server_port=0,
        remote_server_addresses=[
            self._cached_server1_target, self._cached_server2_target,
            self._cached_server3_target
        ],
        task_index=0)

    self.device_local = "/job:%s/replica:0/task:0/device:CPU:0" % JOB_NAME
    self.device_t1 = "/job:%s/replica:0/task:1/device:CPU:0" % JOB_NAME
    self.device_t2 = "/job:%s/replica:0/task:2/device:CPU:0" % JOB_NAME
    self.device_t3 = "/job:%s/replica:0/task:3/device:CPU:0" % JOB_NAME

  def setUp(self):
    super(DynamicClusterTest, self).setUp()
    local_port = pywrap_tfe.TF_PickUnusedPortOrDie()
    context.set_server_def(
        server_def=get_server_def(
            JOB_NAME,
            local_server_port=local_port,
            remote_server_addresses=[
                self._cached_server1_target, self._cached_server2_target
            ],
            task_index=0))

  def tearDown(self):
    super(DynamicClusterTest, self).tearDown()
    ops.device(None).__enter__()
    context._reset_context()

  @test_util.run_in_async_and_sync_mode
  def testServerAdded(self):
    """Add a server to cluster, and run remote ops on it."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    context.update_server_def(server_def=self.server_def_s1_s2_s3)
    with ops.device(self.device_t3):
      x2 = array_ops.ones([2, 2])

    # Test new server accessing resources on old server
    with ops.device(self.device_t3):
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # Test old server accessing resources on new server
    with ops.device(self.device_t2):
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testServerRemoved(self):
    """Remove a server from cluster, and run ops on cluster."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])
    with ops.device(self.device_t2):
      x2 = array_ops.ones([2, 2])

    with ops.device(self.device_t1):
      y = math_ops.matmul(x1, x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    context.update_server_def(server_def=self.server_def_s1)
    with ops.device(self.device_t1):
      y = math_ops.matmul(x1, x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # Running ops on removed server s2 throws an exception
    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device(self.device_t2):
        y = math_ops.matmul(x1, x2)
    self.assertIn("unknown device", cm.exception.message)

    # TODO(haoyuzhang): raise and catch exception when accessing tensors on
    # the removed servers.

  @test_util.run_in_async_and_sync_mode
  def testServerReplaced(self):
    """Replace remote host_port for a task, and run ops on cluster."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    context.update_server_def(server_def=self.server_def_s1_s3)
    with ops.device(self.device_t2):
      y = math_ops.matmul(x1, x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testFunctionServerAdded(self):
    """Add a server to cluster, and run remote function on it."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      return math_ops.matmul(i, i)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    context.update_server_def(server_def=self.server_def_s1_s2_s3)
    with ops.device(self.device_t3):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    with ops.device(self.device_t3):
      x2 = array_ops.ones([2, 2])
    with ops.device(self.device_t1):
      y = worker_fn(x2)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testFunctionServerRemoved(self):
    """Remove a server from cluster, and run ops on cluster."""

    @def_function.function
    def worker_fn(i):
      return math_ops.matmul(i, i)

    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    context.update_server_def(server_def=self.server_def_s1)

    with ops.device(self.device_t1):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # Running functions on removed server s2 throws an exception
    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device(self.device_t2):
        y = worker_fn(x1)
    self.assertIn(" unknown device", cm.exception.message)

    # TODO(haoyuzhang): raise and catch exception when accessing tensors on
    # the removed servers.

  @test_util.run_in_async_and_sync_mode
  def testFunctionServerRemovedAddedBack(self):
    """Add and remove a server, and run functions on cluster."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      return math_ops.matmul(i, i)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    context.update_server_def(server_def=self.server_def_s1_s2_s3)
    with ops.device(self.device_t3):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    context.update_server_def(server_def=self.server_def_s1_s2)
    with ops.device(self.device_t2):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    context.update_server_def(server_def=self.server_def_s1_s2_s3)
    with ops.device(self.device_t3):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testFunctionServerReplaced(self):
    """Replace remote host_port for a task, and run functions on cluster."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      return math_ops.matmul(i, i)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    context.update_server_def(server_def=self.server_def_s1_s3)
    with ops.device(self.device_t2):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testPendingNodesServerReplaced(self):
    """Update cluster when nodes are still pending on remote workers."""
    with ops.device(self.device_local):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      return math_ops.matmul(i, i)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    # Add enough ops so they are pending when changing the cluster
    num_nodes = 10
    ret = [None] * num_nodes
    for i in range(num_nodes):
      with ops.device(self.device_t1):
        ret[i] = worker_fn(x1)
    # While nodes are still pending on worker s1, replace worker s2 with s3.
    context.update_server_def(server_def=self.server_def_s1_s3)
    with ops.device(self.device_t2):
      y = worker_fn(x1)
    for i in range(num_nodes):
      np.testing.assert_array_equal([[2, 2], [2, 2]], ret[i].numpy())
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testMultiThreadPendingNodesServerReplaced(self):
    """Update cluster when other remote function calls are being launched."""
    with ops.device(self.device_local):
      x1 = array_ops.ones([2, 2])

    num_calls = 10
    lock = threading.Lock()

    @def_function.function
    def worker_fn(i):
      return math_ops.matmul(i, i)

    def thread_fn(device, results):
      for i in range(num_calls):
        lock.acquire()
        with ops.device(device):
          y = worker_fn(x1)
        results[i] = y.numpy()
        lock.release()

    def update_server_def_fn():
      for i in range(num_calls):
        lock.acquire()
        context.update_server_def(
            server_def=(self.server_def_s1_s2 if i %
                        2 == 0 else self.server_def_s1_s3))
        lock.release()

    t1_results = [None] * num_calls
    t2_results = [None] * num_calls
    threads = []
    threads.append(threading.Thread(target=thread_fn,
                                    args=(self.device_t1, t1_results)))
    threads.append(threading.Thread(target=thread_fn,
                                    args=(self.device_t2, t2_results)))
    threads.append(threading.Thread(target=update_server_def_fn))
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    for result in t1_results + t2_results:
      np.testing.assert_array_equal([[2, 2], [2, 2]], result)

  @test_util.run_in_async_and_sync_mode
  def testMultiThreadPendingNodesLockFree(self):
    """Update cluster when other remote function calls are being launched."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    num_calls = 10
    self._coord = coordinator.Coordinator()

    @def_function.function
    def worker_fn(i):
      return math_ops.matmul(i, i)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    def thread_fn(device, results):
      for i in range(num_calls):
        with self._coord.stop_on_exception():
          with ops.device(device):
            results[i] = worker_fn(x1).numpy()

    def update_server_def_fn():
      for _ in range(30):
        with self._coord.stop_on_exception():
          context.update_server_def(self.server_def_s1_s2)

    t1_results = [None] * num_calls
    t2_results = [None] * num_calls
    threads = []
    threads.append(
        threading.Thread(target=thread_fn, args=(self.device_t1, t1_results)))
    threads.append(
        threading.Thread(target=thread_fn, args=(self.device_t2, t2_results)))
    threads.append(threading.Thread(target=update_server_def_fn))
    for t in threads:
      t.start()
    self._coord.join(threads)
    for result in t1_results + t2_results:
      np.testing.assert_array_equal([[2, 2], [2, 2]], result)

  @test_util.run_in_async_and_sync_mode
  def testDistributedFunctionServerAdded(self):
    """Add a server to cluster, and run distributed function on it."""
    with ops.device(self.device_t1):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      with ops.device(self.device_t2):
        mul = math_ops.matmul(i, i)
      return mul - array_ops.zeros_like(mul)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    context.update_server_def(server_def=self.server_def_s1_s2_s3)
    with ops.device(self.device_t3):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testDistributedFunctionServerRemovedAddedBack(self):
    """Add then remove a server, and run distributed function on cluster."""
    with ops.device(self.device_local):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      with ops.device(self.device_t1):
        mul = math_ops.matmul(i, i)
      return mul - array_ops.zeros_like(mul)

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    context.update_server_def(server_def=self.server_def_s1)
    with ops.device(self.device_t1):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    context.update_server_def(server_def=self.server_def_s1_s2)
    with ops.device(self.device_t2):
      y = worker_fn(x1)
    np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  @test_util.run_in_async_and_sync_mode
  def testDistributedFunctionBothServersReplaced(self):
    """Tests that replacing servers works correctly.

    We create two servers, t1 and t2. We first replace t2, then we replace t1.

    Among other things, this ensures that both already existing, and
    restarted workers have the context view IDs correctly updated.
    """
    with ops.device(self.device_local):
      x1 = array_ops.ones([2, 2])

    @def_function.function
    def worker_fn(i):
      with ops.device(self.device_t1):
        mul = math_ops.matmul(i, i)
      with ops.device(self.device_t2):
        add = mul + i
      return add - i

    # Forces function tracing and registration
    worker_fn.get_concrete_function(x1)

    # Replace task2
    context.update_server_def(server_def=self.server_def_s1_s3)
    for device in (self.device_t1, self.device_t2):
      with ops.device(device):
        y = worker_fn(x1)
      np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

    # Then replace task1
    context.update_server_def(server_def=self.server_def_s4_s3)
    for device in (self.device_t1, self.device_t2):
      with ops.device(device):
        y = worker_fn(x1)
      np.testing.assert_array_equal([[2, 2], [2, 2]], y.numpy())

  def testCheckAlive(self):
    with self.assertRaisesRegexp(ValueError, "Context is not initialized."):
      context.check_alive("/job:remote_device/task:0")
    context.context().ensure_initialized()

    self.assertTrue(context.check_alive("/job:remote_device/replica:0/task:0"))
    self.assertTrue(context.check_alive("/job:remote_device/replica:0/task:1"))

    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "Client for target /job:remote_device/replica:0/task:10 not found."):
      context.check_alive("/job:remote_device/replica:0/task:10")


class DynamicClusterWithoutLazyRemoteInputsCopyTest(DynamicClusterTest):

  @classmethod
  def setUpClass(cls):
    super(DynamicClusterWithoutLazyRemoteInputsCopyTest, cls).setUpClass()
    context._reset_context()
    context.context().lazy_remote_inputs_copy = False

  @classmethod
  def tearDownClass(cls):
    super(DynamicClusterWithoutLazyRemoteInputsCopyTest, cls).tearDownClass()
    context._reset_context()
    context.context().lazy_remote_inputs_copy = True

  # TODO(haoyuzhang): When lazyh remote inputs copy is disabled, we use the
  # WorkerService RunGraph request to execute component functions in distributed
  # function execution. We currently do not have access control in WorkerService
  # to allow concurrent cluster update and function execution.
  def testMultiThreadPendingNodesLockFree(self):
    self.skipTest("Unsupported case")


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
