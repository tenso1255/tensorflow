/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_POPLAR_DRIVER_POPLAR_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_POPLAR_DRIVER_POPLAR_EXECUTABLE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"

#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {

// A Poplar executable is a wrapper around and Engine, with
// the execution Sequence program, input tensors and output
// tensor recorded.
class PoplarExecutable : public Executable {
 public:
  PoplarExecutable(std::unique_ptr<HloModule> hlo_module,
                   std::unique_ptr<poplar::Engine> engine,
                   const std::map<int64, int64>& output_map);
  ~PoplarExecutable() override;


  StatusOr<perftools::gputools::DeviceMemoryBase> ExecuteOnStream(
          const ServiceExecutableRunOptions* run_options,
          tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments,
          HloExecutionProfile* hlo_execution_profile) override;

  StatusOr<std::unique_ptr<ShapedBuffer>> ExecuteOnStream(
          const ServiceExecutableRunOptions* run_options,
          tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
          HloExecutionProfile* hlo_execution_profile) override;

  StatusOr<perftools::gputools::DeviceMemoryBase> ExecuteAsyncOnStream(
          const ServiceExecutableRunOptions* run_options,
          tensorflow::gtl::ArraySlice<perftools::gputools::DeviceMemoryBase>
          arguments) override;

  static int64 ShapeSizeBytes(const Shape& shape);

 private:
  std::unique_ptr<poplar::Engine> poplar_engine_;
  std::map<int64, int64> output_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(PoplarExecutable);
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_POPLAR_DRIVER_POPLAR_EXECUTABLE_H_
