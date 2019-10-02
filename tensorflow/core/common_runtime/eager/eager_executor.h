/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_EXECUTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_EXECUTOR_H_

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class AsyncEagerNode;

// A unit of execution for the EagerExecutor class below. Example subclasses
// encapsulate execution of a TFE_Op, or copying a TFE_TensorHandle from one
// device to another.
class EagerNode {
 public:
  EagerNode() {}

  virtual ~EagerNode() {}

  // Prepares the node when adding it into EagerExecutor. If any errors happens,
  // EagerExecutor will abort the node immediately.
  virtual Status Prepare() { return Status::OK(); }

  // Runs the computation corresponding to this node and blocks till the
  // execution is done.
  virtual Status Run() = 0;

  // Called when this node will not be run due to some error contained in
  // `status`. `status` must not be OK.
  // For example, if the node would have computed some tensors in the Run(),
  // it should poison the corresponding tensor handles in this method.
  virtual void Abort(Status status) = 0;

  // Returns nullptr iff this Eager node is synchronous.
  virtual AsyncEagerNode* AsAsync() { return nullptr; }

  virtual string DebugString() const = 0;
};

class AsyncEagerNode : public EagerNode {
 public:
  using EagerNode::EagerNode;  // Lift EagerNode constructors.

  // This node will be cleaned up once the done callback is called.
  virtual void RunAsync(StatusCallback done) = 0;

  AsyncEagerNode* AsAsync() final { return this; }

  Status Run() final {
    return errors::Unimplemented("Don't call AsyncEagerNode::Run().");
  }
};

// A class for handling async execution (see TFE_ContextSetAsync).
// Note that this class is thread-safe.
// TODO(agarwal): TFE_OpAddInput may currently block if it tries to access the
// device of the input handle. Fix that.
// TODO(agarwal): Implement support for control dependencies.
// TODO(agarwal): Support out-of-order execution and dispatching multiple
// EagerNode in parallel.
// TODO(agarwal): Implement optimizations over EagerNode traces.
class EagerExecutor {
 public:
  explicit EagerExecutor(bool async);

  ~EagerExecutor();

  // Puts this in a shutdown state. In this state, Add() will return an error
  // and not add new EagerNodes. After putting this in the shutdown state,
  // blocks until all pendings nodes have finished running.
  // Returns the status of executing pending nodes.
  // If async was not enabled, aborts and destroys all pending nodes.
  Status ShutDown();

  bool Async() const;

  // - Async Mode: schedules `node` for execution.
  // - Sync Mode: inline execute the 'node' directly.
  // If an error occurs (e.g. EagerExecutor has already been shut down), the
  // `node` is not added to this executor and its Abort() method is called.
  Status AddOrExecute(std::unique_ptr<EagerNode> node);

  // Blocks till all currently pending ops are done.
  // In particular, if EnableAsync() has not beed called, it will not return
  // until that happens (and pendings, at the time of call, nodes finish
  // running). If this executor has already been shut down, its final status is
  // returned.
  Status WaitForAllPendingNodes();

  // Clears all currently set errors which re-enables async execution.
  void ClearError();

  // Returns Status based on any errors that occurred during async execution.
  Status status() const;

 private:
  // Possible states for this executor.
  // Executor starts in kActive state. When Shutdown() is called, Executor
  // is put in the kShuttingDown state. In this state, the executor thread
  // continues to run, but no new nodes are accepted. Finally, when all nodes
  // are drained, the executor is put in the kShutDown state, which causes the
  // thread to exit.
  // If this executor is destroyed without calling shutdown first, it
  // transitions to kShutDown state immediately which causes the thread to exit
  // without running pending nodes.
  enum class ExecutorState {
    kActive,
    kShuttingDown,
    kShutDown,
  };

  enum class NodeState {
    kPENDING,
    kSCHEDULED,
    kDONE,
  };

  struct NodeItem : core::RefCounted {
    // Unique id generated in EagerExecutor::Add(). If item1.id < item2.id, it
    // means item1.node is added before item2.node.
    uint64 id;
    std::unique_ptr<EagerNode> node;
    NodeState state;
  };

  const char* StateStringLocked() EXCLUSIVE_LOCKS_REQUIRED(node_queue_mutex_);

  void NodeDone(core::RefCountPtr<NodeItem> item, const Status& status);

  // Starts execution of pending EagerNodes. This function loops till
  // thread_done_ is set to true. If any errors are encontered, these are set
  // inside `status_`. The loop blocks anytime there are no pending nodes, or if
  // `status_` is not ok.
  void Run();

  void RunItem(core::RefCountPtr<NodeItem> item);

  // The impl of WaitForAllPendingNodes
  // `lock` is the lock that holds node_queue_mutex_.
  Status WaitForAllPendingNodesLocked(mutex_lock* lock)
      EXCLUSIVE_LOCKS_REQUIRED(node_queue_mutex_);

  Status WaitImpl(bool wait_all, uint64 node_id);

  std::atomic<uint64> next_node_id_;

  mutable mutex node_queue_mutex_;

  // Used to signal that some EagerNodes are pending execution.
  condition_variable nodes_pending_ GUARDED_BY(node_queue_mutex_);

  // Queue of pending NodeItems. Ordered by NodeItem::id.
  std::queue<core::RefCountPtr<NodeItem>> node_queue_
      GUARDED_BY(node_queue_mutex_);

  // Ordered by NodeItem::id.
  std::map<uint64, core::RefCountPtr<NodeItem>, std::less<uint64>>
      unfinished_nodes_ GUARDED_BY(node_queue_mutex_);

  // `status_` is set based on any errors raised during execution of a
  // EagerNode.  It remains set until ClearError is called.
  Status status_ GUARDED_BY(node_queue_mutex_);

  // Map from id of a EagerNode to condition_variables (not owned by the map).
  // These condition_variables are notified and removed when that EagerNode is
  // done executing, or if an error is found in execution of any EagerNode.
  // The map is ordered by id.
  std::multimap<uint64, condition_variable*, std::less<uint64>>
      node_done_notifications_ GUARDED_BY(node_queue_mutex_);

  // thread_exited_notification_ is notified by the `thread_` right before it
  // exits.
  Notification thread_exited_notification_;

  // Indicates that `thread_` should stop as soon as it is done executing the
  // current EagerNode.
  ExecutorState state_ GUARDED_BY(node_queue_mutex_) = ExecutorState::kActive;

  // Thread object that calls the `Run` method in async mode.This thread runs
  // until state_ is set to kShuttingDown. It is `nullptr` in sync mode.
  const std::unique_ptr<Thread> thread_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_EXECUTOR_H_
