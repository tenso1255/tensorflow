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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_RESOURCES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_COMPILER_RESOURCES_H_

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/graph_caching_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_subcomputation.h"

#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poprand/RandomGen.hpp>
#include <poputil/GraphFunction.hpp>

namespace xla {
namespace poplarplugin {

using ComputationMap = std::map<const HloComputation*, SubComputationVisitor>;

// This structure contains additional information required to lower the graph
// from an XLA graph to a poplar graph.
struct CompilerResources {
  ComputationMap computation_map;

  CompilerAnnotations annotations;

  poplin::PlanningCache convolution_cache;

  poplin::matmul::PlanningCache dot_cache;

  poprand::Random random;

  graph_caching_util::ConvolutionGraphCache conv_graph_cache;

  graph_caching_util::BwdWeightGraphCache bwd_weight_graph_cache;

  graph_caching_util::WeightUpdateConvolutionGraphCache wu_graph_cache;

  CompilerResources(uint64 seed, poprand::RandomGenMode mode)
      : random(mode, seed) {}
};

}  // namespace poplarplugin
}  // namespace xla

#endif
