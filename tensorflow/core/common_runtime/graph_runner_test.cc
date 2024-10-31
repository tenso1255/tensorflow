/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/graph_runner.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

TEST(GraphRunnerTest, SingleConst) {
  Scope root = Scope::NewRootScope();
  auto c = ops::Const(root, 42.0f);
  GraphRunner graph_runner(Env::Default());
  std::vector<Tensor> outputs;
  absl::Status s =
      graph_runner.Run(root.graph(), nullptr, {}, {c.name()}, &outputs);
  TF_ASSERT_OK(s);
  test::ExpectEqual(test::AsScalar(42.0f), outputs[0]);
}

// If not using DeepCopy, and the allocator is deleted with the cpu-device,
// this test will seg-fault.
TEST(GraphRunnerTest, DeepCopy) {
  Scope root = Scope::NewRootScope();
  auto p1 = ops::Placeholder(root.WithOpName("p1"), DT_FLOAT);
  auto p2 = ops::Placeholder(root.WithOpName("p2"), DT_FLOAT);
  auto add = ops::Add(root.WithOpName("add"), p1, p2);

  Tensor p1_data(DT_FLOAT, TensorShape({}));
  Tensor p2_data(DT_FLOAT, TensorShape({}));
  p1_data.scalar<float>()() = 1.0f;
  p2_data.scalar<float>()() = 2.0f;
  std::vector<std::pair<string, Tensor>> inputs = {{"p1:0", p1_data},
                                                   {"p2:0", p2_data}};

  // Create and destroy the GraphRunner, and ensure that the outputs are
  // consumable beyond the lifetime of GraphRunner.
  std::vector<Tensor> outputs;
  {
    GraphRunner graph_runner(Env::Default());
    absl::Status s =
        graph_runner.Run(root.graph(), nullptr, inputs, {"add:0"}, &outputs);
    TF_ASSERT_OK(s);
  }
  test::ExpectEqual(test::AsScalar(3.0f), outputs[0]);
}

TEST(GraphRunnerTest, MultiFetchConst) {
  Scope root = Scope::NewRootScope();
  auto c = ops::Const(root, 42.0f);
  auto pi = ops::Const(root, 3.14f);
  GraphRunner graph_runner(Env::Default());
  std::vector<Tensor> outputs;
  absl::Status s = graph_runner.Run(root.graph(), nullptr, {},
                                    {c.name(), pi.name()}, &outputs);
  TF_ASSERT_OK(s);
  test::ExpectEqual(test::AsScalar(42.0f), outputs[0]);
  test::ExpectEqual(test::AsScalar(3.14f), outputs[1]);
}

TEST(GraphRunnerTest, FeedAndFetch) {
  Scope root = Scope::NewRootScope();
  auto p1 = ops::Placeholder(root.WithOpName("p1"), DT_FLOAT);
  auto p2 = ops::Placeholder(root.WithOpName("p2"), DT_FLOAT);
  auto add = ops::Add(root.WithOpName("add"), p1, p2);

  Tensor p1_data(DT_FLOAT, TensorShape({}));
  Tensor p2_data(DT_FLOAT, TensorShape({}));
  p1_data.scalar<float>()() = 1.0f;
  p2_data.scalar<float>()() = 2.0f;
  std::vector<std::pair<string, Tensor>> inputs = {{"p1:0", p1_data},
                                                   {"p2:0", p2_data}};

  GraphRunner graph_runner(Env::Default());
  std::vector<Tensor> outputs;
  absl::Status s =
      graph_runner.Run(root.graph(), nullptr, inputs, {"add:0"}, &outputs);
  TF_ASSERT_OK(s);
  test::ExpectEqual(test::AsScalar(3.0f), outputs[0]);
}

}  // namespace
}  // namespace tensorflow
