/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_xla_computations_pass.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/lower_functional_ops.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/partitioning_utils.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace tfrt_stub {

namespace {

// Finds the names of functions that are safe to optimize.
absl::flat_hash_set<std::string> FindFunctionsToOptimize(
    const GraphDef& graph_def) {
  // TODO(b/203689805): Add more functional ops.
  static const auto* const kOpWhitelist = new absl::flat_hash_set<std::string>{
      "PartitionedCall", "StatefulPartitionedCall"};
  absl::flat_hash_map<
      std::string /*function_name*/,
      absl::flat_hash_set<std::string> /*ops_using_the_function*/>
      function_to_ops;

  auto build_map = [&](const auto& node_defs) {
    for (const auto& node_def : node_defs) {
      for (const auto& p : node_def.attr()) {
        const AttrValue& attr_value = p.second;
        if (!attr_value.has_func()) continue;
        function_to_ops[attr_value.func().name()].insert(node_def.op());
      }
    }
  };

  build_map(graph_def.node());
  for (const auto& function_def : graph_def.library().function()) {
    build_map(function_def.node_def());
  }

  absl::flat_hash_set<std::string> functions_to_optimize;
  for (const auto& p : function_to_ops) {
    const std::string& function_name = p.first;
    const absl::flat_hash_set<std::string>& ops = p.second;
    // Optimize a function iff all the ops that use it are whitelisted.
    if (std::all_of(ops.begin(), ops.end(), [](const auto& op) {
          return kOpWhitelist->contains(op);
        })) {
      functions_to_optimize.insert(function_name);
    }
  }

  return functions_to_optimize;
}

// Preprocesses `graph_def`, returns the functions to optimize if
// `run_placer_grappler_on_functions` is true.
StatusOr<absl::flat_hash_set<std::string>> PreprocessGraph(
    tensorflow::GraphDef& graph_def, bool run_placer_grappler_on_functions) {
  if (VLOG_IS_ON(1)) {
    DumpGraphDefToFile("before_generate_resource_shared_name_graph_def",
                       graph_def);
  }

  TF_RETURN_IF_ERROR(tensorflow::GenerateResourceSharedNameIfEmpty(
      graph_def, tensorflow::OpRegistry::Global()));

  if (VLOG_IS_ON(2)) {
    DumpGraphDefToFile("after_generate_resource_shared_name_graph_def",
                       graph_def);
  }

  if (run_placer_grappler_on_functions) {
    return FindFunctionsToOptimize(graph_def);
  }
  return absl::flat_hash_set<std::string>();
}

}  // namespace

StatusOr<std::unique_ptr<TfrtGraphExecutionState>>
TfrtGraphExecutionState::Create(const TfrtGraphExecutionState::Options& options,
                                tensorflow::GraphDef graph_def,
                                const FallbackState& fallback_state) {
  TF_ASSIGN_OR_RETURN(
      auto functions_to_optimize,
      PreprocessGraph(graph_def, options.run_placer_grappler_on_functions));

  // `CreateGraphExecutionState()` will preprocess the graph (e.g., apply
  // Placer to the top level graph).
  TF_ASSIGN_OR_RETURN(
      auto graph_execution_state,
      fallback_state.CreateGraphExecutionState(std::move(graph_def)));

  return std::make_unique<TfrtGraphExecutionState>(
      options, std::move(graph_execution_state), fallback_state,
      std::move(functions_to_optimize));
}

namespace {

CallableOptions PopulateCallableOptions(
    CallableOptions& callable_options,
    absl::Span<const std::string> feed_tensor_names,
    absl::Span<const std::string> fetch_tensor_names,
    absl::Span<const std::string> target_tensor_names) {
  // Configure pruning with the feed/fetch/target tensor names.
  callable_options.mutable_feed()->Reserve(feed_tensor_names.size());
  for (const auto& feed : feed_tensor_names) {
    callable_options.add_feed(feed);
  }
  callable_options.mutable_fetch()->Reserve(fetch_tensor_names.size());
  for (const auto& fetch : fetch_tensor_names) {
    callable_options.add_fetch(fetch);
  }
  callable_options.mutable_target()->Reserve(target_tensor_names.size());
  for (const auto& target : target_tensor_names) {
    callable_options.add_target(target);
  }

  return callable_options;
}

tensorflow::GraphDef CreateGraphDefFromGraphAndFlibDef(
    const tensorflow::Graph& graph,
    const tensorflow::FunctionLibraryDefinition& flib_def) {
  tensorflow::GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  *graph_def.mutable_library() = flib_def.ToProto();
  return graph_def;
}

// Creates a pruned graph from `graph_def` according to `callable_options`.
StatusOr<std::unique_ptr<tensorflow::Graph>> CreatePrunedGraph(
    tensorflow::GraphDef graph_def, const CallableOptions& callable_options) {
  VLOG(1) << "Creating pruned graph: " << callable_options.DebugString();

  // Prune the graph with `callable_options`. Although
  // grappler has model_pruner stage, it may leave v1 control flows in an
  // invalid state that cannot be functionalized. So we perform additional
  // pruning before functionalization.
  TF_RETURN_IF_ERROR(PruneGraphDef(graph_def, callable_options));

  if (VLOG_IS_ON(2)) {
    DumpGraphDefToFile("before_eliminate_ref_variables_graph_def", graph_def);
  }

  // Ref variables in V1 Control flow prevent it from being functionalized. So
  // we eliminate them first.
  TF_RETURN_IF_ERROR(EliminateRefVariablesFromV1ControlFlow(graph_def));

  // The "_input_shapes" attributes will be not be correct after function
  // optimizer in grappler, we need to remove them. Note that "_input_shapes" is
  // not used except as a debug hint (somehow this debug hint is used by MLIR
  // graphdef importer, which is not expected).
  RemoveInputShapesInFunctions(graph_def);

  auto pruned_graph =
      std::make_unique<tensorflow::Graph>(tensorflow::OpRegistry::Global());
  tensorflow::GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.add_default_attributes = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(options, std::move(graph_def),
                                            pruned_graph.get()));
  return pruned_graph;
}

// Creates a new identity node to replace an operand of a given `node`.
NodeDef CreateNewIdentityNode(const NodeDef& node,
                              const std::string& input_name,
                              const std::string& identity_name) {
  NodeDef identity;
  identity.set_name(identity_name);
  identity.set_op("Identity");
  identity.add_input(input_name);
  identity.set_device(node.device());
  for (const auto& name_and_attr : node.attr()) {
    if (name_and_attr.first == "T") {
      identity.mutable_attr()->insert(name_and_attr);
      break;
    }
  }
  return identity;
}

// Inlines functions into the top level graph.
Status InlineFunctions(std::unique_ptr<Graph>* graph,
                       const DeviceSet* device_set) {
  GraphOptimizationPassOptions optimization_options;
  SessionOptions session_options;
  // We don't lower v2 control flow to v1 for now.
  session_options.config.mutable_experimental()->set_use_tfrt(true);
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);
  optimization_options.session_options = &session_options;
  optimization_options.graph = graph;
  optimization_options.flib_def = (*graph)->mutable_flib_def();
  optimization_options.device_set = device_set;
  optimization_options.is_function_graph = false;

  LowerFunctionalOpsPass pass;
  return pass.Run(optimization_options);
}

// Assigns input/output nodes to the host.
Status PlaceInputOutputNodesOnHost(const std::vector<std::string>& inputs,
                                   const std::vector<std::string>& outputs,
                                   const Device* cpu_device, Graph* graph) {
  std::unordered_map<std::string, Node*> name_to_node_map =
      graph->BuildNodeNameIndex();
  for (const auto& input : inputs) {
    name_to_node_map.at(grappler::NodeName(input))
        ->set_assigned_device_name(cpu_device->name());
  }

  // Collect all output nodes.
  absl::flat_hash_set<Node*> output_nodes;
  for (const auto& output : outputs) {
    output_nodes.insert(name_to_node_map.at(grappler::NodeName(output)));
  }
  for (const auto& output_node : output_nodes) {
    // Append an IdentityN node to the original output node if it is not
    // assigned to the host.
    if (!output_node->IsIdentity() &&
        output_node->type_string() != "IdentityN" &&
        output_node->assigned_device_name() != cpu_device->name()) {
      // Rename the original output node.
      std::string output_node_name = output_node->name();
      output_node->set_name(output_node_name + "/tfrt_renamed");

      // Append an IdentityN node with the original output node name.
      std::vector<NodeBuilder::NodeOut> output_tensors;
      output_tensors.reserve(output_node->num_outputs());
      for (int i = 0; i < output_node->num_outputs(); i++) {
        output_tensors.push_back(NodeBuilder::NodeOut(output_node, i));
      }
      TF_RETURN_IF_ERROR(NodeBuilder(output_node_name, "IdentityN")
                             .AssignedDevice(cpu_device->name())
                             .Input(output_tensors)
                             .Finalize(graph, /*created_node=*/nullptr));
    } else {
      output_node->set_assigned_device_name(cpu_device->name());
    }
  }
  return OkStatus();
}

Status AdjustDeviceAssignment(const std::vector<std::string>& inputs,
                              const std::vector<std::string>& outputs,
                              const std::vector<std::string>& control_outputs,
                              const Device* cpu_device, Graph* graph) {
  // TODO(b/232299232): We don't inline and partition v2 control flow currently.
  // All ops within control flow are placed on CPU for now. Figure out a better
  // way to handle v2 control flow.
  for (Node* node : graph->op_nodes()) {
    if (node->IsWhileNode() || node->IsIfNode()) {
      LOG(WARNING) << "The control flow node " << node->name()
                   << " is placed on CPU.";
      node->set_assigned_device_name(cpu_device->name());
    }
  }

  TF_RETURN_IF_ERROR(
      PlaceInputOutputNodesOnHost(inputs, outputs, cpu_device, graph));
  return OkStatus();
}

bool IsTpuGraph(const Graph* graph) {
  static const auto* const kTpuOps = new absl::flat_hash_set<std::string>{
      "TPUPartitionedCall", "TPUCompile", "TPUReplicateMetadata"};
  for (const Node* node : graph->nodes()) {
    if (kTpuOps->contains(node->type_string())) {
      return true;
    }
  }
  for (const std::string& func_name : graph->flib_def().ListFunctionNames()) {
    const FunctionDef* func_def = graph->flib_def().Find(func_name);
    for (const NodeDef& node_def : func_def->node_def()) {
      if (kTpuOps->contains(node_def.op())) return true;
    }
  }
  return false;
}

// Adds Send/Recv ops to `graph` for data transfer, if ops are run on different
// devices. Returns a new graph with the added Send/Recv ops.
// This is done by partitioning `graph` and add Send/Recv ops on the edges
// across devices.
StatusOr<std::unique_ptr<Graph>> BuildXlaOpsAndMaybeInsertTransferOps(
    const std::string& graph_func_name, const FallbackState& fallback_state,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& control_outputs,
    std::unique_ptr<Graph> graph) {
  // Skip inserting transfer ops if this is a TPU graph.
  // Our stack currently cannot run the old bridge on TPU graphs, as it will
  // generate ops that are not supported by the subsequent MLIR passes.
  // In the case where TPU related ops are not wrapped in TPUPartitionedCall,
  // running placer and partitioning on such graphs will fail. So we skip TPU
  // graphs for now.
  // TODO(b/228510957): In the long term, we will want a unified way for data
  // transfer, i.e., using Send/Recv ops for data transfer for TPU as well.
  if (IsTpuGraph(graph.get())) {
    return graph;
  }

  // Inline functions to facilitate partitioning nodes in the functions.
  TF_RETURN_IF_ERROR(InlineFunctions(&graph, &fallback_state.device_set()));
  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("after_inlining", *graph);
  }

  // Replace the StatefulPartitionedCall op that should be compiled to an
  // XlaLaunch op.
  // TODO(b/239089915): Clean this up after the logic is implemented in TFXLA
  // bridge.
  TF_RETURN_IF_ERROR(BuildXlaLaunchOps(graph.get()));
  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("after_build_xla_launch", *graph);
  }

  // Run placer.
  const Device* cpu_device = fallback_state.device_manager().HostCPU();
  if (cpu_device == nullptr) {
    return errors::Internal("No CPU device found.");
  }
  Placer placer(graph.get(), /*function_name=*/"", &graph->flib_def(),
                &fallback_state.device_set(), cpu_device,
                /*allow_soft_placement=*/true,
                /*log_device_placement=*/false);
  TF_RETURN_IF_ERROR(placer.Run());
  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("after_placer", *graph);
  }

  TF_RETURN_IF_ERROR(AdjustDeviceAssignment(inputs, outputs, control_outputs,
                                            cpu_device, graph.get()));

  // Insert send/recv ops to the graph.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Graph> new_graph,
      InsertTransferOps(fallback_state.device_set(), std::move(graph)));
  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("after_transfer_ops_insertion", *new_graph);
  }

  return new_graph;
}

}  // namespace

StatusOr<TfrtGraphExecutionState::OptimizationResult>
TfrtGraphExecutionState::CreateOptimizedGraph(
    tensorflow::GraphImportConfig& graph_import_config) {
  OptimizationResult result;

  tensorflow::BuildGraphOptions build_graph_options;

  std::vector<std::string> inputs;
  inputs.reserve(graph_import_config.inputs.size());
  for (const auto& input : graph_import_config.inputs) {
    inputs.push_back(input.first);
  }
  PopulateCallableOptions(build_graph_options.callable_options, inputs,
                          graph_import_config.outputs,
                          graph_import_config.control_outputs);

  auto graph_def = CreateGraphDefFromGraphAndFlibDef(graph(), flib_def());

  if (VLOG_IS_ON(1)) {
    DumpGraphDefToFile("before_pruning", graph_def);
  }

  TF_ASSIGN_OR_RETURN(
      result.graph,
      CreatePrunedGraph(graph_def, build_graph_options.callable_options));
  DCHECK(result.graph);

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("after_pruning", *result.graph);
  }

  const auto functionalization_start_time = absl::Now();

  // Perform functionalization to convert v1 control flow to v2 control flow. It
  // should be applied to the unoptimized graph, because Grappler may cause
  // unfunctionalizablity.
  TF_RETURN_IF_ERROR(tensorflow::UpgradeLegacyGraph(
      result.graph.get(),
      const_cast<tensorflow::FunctionLibraryDefinition*>(
          &result.graph->flib_def()),
      /*restrict_functionalization_to_compiled_nodes=*/false));

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("after_functionalization", *result.graph);
  }

  auto grappler_start_time = absl::Now();
  result.functionalization_duration =
      grappler_start_time - functionalization_start_time;

  auto status_or_optimized_graph =
      OptimizeGraph(*result.graph, build_graph_options);
  if (status_or_optimized_graph.ok()) {
    result.graph = std::move(status_or_optimized_graph.value());
  } else {
    LOG(WARNING) << "TFRT failed to optimize graph: "
                 << status_or_optimized_graph.status();
  }

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("after_grappler", *result.graph);
  }

  result.grappler_duration = absl::Now() - grappler_start_time;

  if (options_.enable_tfrt_gpu && !options_.use_bridge_for_gpu) {
    TF_ASSIGN_OR_RETURN(
        result.graph,
        BuildXlaOpsAndMaybeInsertTransferOps(
            graph_import_config.graph_func_name, fallback_state_, inputs,
            graph_import_config.outputs, graph_import_config.control_outputs,
            std::move(result.graph)));

    // Update `control_outputs` as there might be newly added Send ops.
    for (const Node* node : result.graph->nodes()) {
      if (node->IsSend()) {
        graph_import_config.control_outputs.push_back(node->name());
      }
    }
  }

  return result;
}

Status TfrtGraphExecutionState::Extend(const GraphDef& graph) {
  std::unique_ptr<GraphExecutionState> new_state;
  absl::MutexLock lock(&graph_execution_state_mu_);
  TF_RETURN_IF_ERROR(graph_execution_state_->Extend(graph, &new_state));
  graph_execution_state_.swap(new_state);

  auto* graph_def = graph_execution_state_->original_graph_def();
  DCHECK_NE(graph_def, nullptr);
  TF_ASSIGN_OR_RETURN(
      functions_to_optimize_,
      PreprocessGraph(*graph_def, options_.run_placer_grappler_on_functions));

  return OkStatus();
}

namespace {

// Given an "Exit" node, finds its corresponding "LoopCond" node.
StatusOr<const NodeDef*> FindLoopCondFromExitNode(
    const NodeDef& exit_node,
    const absl::flat_hash_map<std::string, NodeDef*>& name_to_node) {
  const NodeDef* switch_node = nullptr;
  for (const std::string& tensor_name : exit_node.input()) {
    const std::string node_name = grappler::NodeName(tensor_name);
    if (!name_to_node.contains(node_name)) {
      return errors::InvalidArgument("Graph does not contain input ", node_name,
                                     " of exit node ", exit_node.name());
    }
    const NodeDef* node = name_to_node.at(node_name);
    if (node->op() == "Switch") {
      switch_node = node;
      break;
    }
  }
  if (switch_node == nullptr) {
    return errors::InvalidArgument("Exit node ", exit_node.name(),
                                   " does not have a Switch node as its ",
                                   "predecessor.");
  }
  for (const std::string& tensor_name : switch_node->input()) {
    const std::string node_name = grappler::NodeName(tensor_name);
    if (!name_to_node.contains(node_name)) {
      return errors::InvalidArgument("Graph does not contain input ", node_name,
                                     " of switch node ", switch_node->name());
    }

    const NodeDef* node = name_to_node.at(node_name);
    if (node->op() == "LoopCond") {
      return node;
    }
  }

  return errors::InvalidArgument("Switch node ", switch_node->name(),
                                 " does not have a LoopCond node as its ",
                                 "predecessor.");
}

}  // namespace

Status PruneGraphDef(GraphDef& graph_def,
                     const CallableOptions& callable_options) {
  // Gather node names and create a map from names to NodeDefs.
  absl::flat_hash_map<std::string, NodeDef*> name_to_node;
  // All exit nodes in order to track all while loops.
  absl::flat_hash_set<const NodeDef*> exit_nodes;
  for (auto& node : *graph_def.mutable_node()) {
    name_to_node[node.name()] = &node;
    if (node.op() == "Exit") {
      exit_nodes.insert(&node);
    }

    // TODO(tfrt-devs): Add support for _Send and _Recv ops.
    if (node.op() == "_Send" || node.op() == "_Recv") {
      return errors::InvalidArgument(
          "TFRT prune graphdef cannot handle graphs contains _Send and _Recv "
          "ops.");
    }
  }

  // Find all LoopCond -> Exit nodes mapping. So when we traverse to a LoopCond
  // node, we can add corresponding Exit nodes to the traversal queue in order
  // to maintain complete structure of a while loop.
  absl::flat_hash_map<const NodeDef*, absl::flat_hash_set<const NodeDef*>>
      loop_cond_to_exit_nodes;
  for (const NodeDef* exit_node : exit_nodes) {
    TF_ASSIGN_OR_RETURN(const NodeDef* loop_cond_node,
                        FindLoopCondFromExitNode(*exit_node, name_to_node));
    loop_cond_to_exit_nodes[loop_cond_node].insert(exit_node);
  }

  // `queue` is for candidate nodes we want to visit in the graph.
  std::vector<const NodeDef*> queue;

  // Add fetch nodes to the queue.
  absl::flat_hash_set<std::string> fetch_node_names;
  for (const std::string& tensor_name : callable_options.fetch()) {
    const NodeDef* node = name_to_node[grappler::NodeName(tensor_name)];
    if (!node) {
      return errors::InvalidArgument("Graph does not contain fetch node ",
                                     tensor_name, ".");
    }
    queue.push_back(node);
    fetch_node_names.insert(node->name());
  }

  // Add control target nodes to the queue.
  for (const std::string& tensor_name : callable_options.target()) {
    const NodeDef* node = name_to_node[grappler::NodeName(tensor_name)];
    if (!node) {
      return errors::InvalidArgument("Graph does not contain target node ",
                                     tensor_name, ".");
    }
    queue.push_back(node);
    fetch_node_names.insert(node->name());
  }

  absl::flat_hash_set<NodeDef*> feed_node_defs;

  // Add feed nodes to the queue. In addition, perform necessary rewrites to
  // remove unnecessary input edges.
  for (const std::string& tensor_name : callable_options.feed()) {
    NodeDef* node = name_to_node[grappler::NodeName(tensor_name)];
    if (!node) {
      return errors::InvalidArgument("Graph does not contain feed node ",
                                     tensor_name, ".");
    }

    // If a feed node is a Const, we don't need its inputs at all.
    //
    // TODO(tfrt-devs): Consider a general solution that we could just rewrite
    // all feed nodes to Placeholder nodes.
    if (node->op() == "Const") {
      node->clear_input();
    }

    queue.push_back(node);
    feed_node_defs.insert(node);
  }

  absl::flat_hash_set<const NodeDef*> visited;
  std::vector<NodeDef> keep;

  // Perform graph traversal to find out connected nodes from fetches.
  while (!queue.empty()) {
    const NodeDef* node = queue.back();
    queue.pop_back();

    if (!visited.insert(node).second) {
      continue;
    }

    keep.push_back(*node);
    if (node->op() == "LoopCond") {
      for (const NodeDef* exit_node : loop_cond_to_exit_nodes[node]) {
        queue.push_back(exit_node);
      }
    }

    for (const std::string& tensor_name : node->input()) {
      const NodeDef* in = name_to_node[grappler::NodeName(tensor_name)];
      if (!in) {
        return errors::InvalidArgument("Graph does not contain input ",
                                       grappler::NodeName(tensor_name),
                                       " of node ", node->name(), ".");
      }
      queue.push_back(in);
    }
  }

  graph_def.clear_node();
  for (auto& node : keep) {
    if (fetch_node_names.contains(node.name())) {
      // If the fetch node is an Exit op, we insert an Identity op right after
      // it and rename it to be the new fetch node. This is to prevent
      // functionalization from removing the fetch nodes.
      if (node.op() == "Exit") {
        auto renamed_exit_node = node;
        renamed_exit_node.set_name(
            absl::StrCat(renamed_exit_node.name(), "/tfrt_renamed"));
        node.set_op("Identity");
        *node.mutable_input(0) = renamed_exit_node.name();
        *graph_def.add_node() = std::move(renamed_exit_node);
      }
    }

    *graph_def.add_node() = std::move(node);
  }

  return OkStatus();
}

Status EliminateRefVariablesFromV1ControlFlow(tensorflow::GraphDef& graph_def) {
  auto* op_factory = OpRegistry::Global();

  absl::flat_hash_set<std::string> ref_nodes;
  for (const auto& node : graph_def.node()) {
    if (node.op() == "RefEnter" || node.op() == "RefSwitch") {
      ref_nodes.insert(node.name());
    }
  }

  tensorflow::GraphDef updated_graph_def;
  absl::flat_hash_set<std::string> new_identities;
  // Insert an identity node between each "RefEnter" or "RefSwitch" node and its
  // ref input. Then modify each "RefEnter"/"RefSwitch" node in-place to an
  // "Enter"/"Switch" node.
  for (auto& node : *graph_def.mutable_node()) {
    // First find the ref input name to this RefEnter or RefSwitch.
    std::string* ref_input_name = nullptr;
    if (node.op() == "RefEnter") {
      node.set_op("Enter");
      if (node.input_size() != 1) {
        return errors::InvalidArgument("RefEnter node ", node.name(),
                                       " does not have exactly 1 input.");
      }
      ref_input_name = node.mutable_input(0);
    } else if (node.op() == "RefSwitch") {
      node.set_op("Switch");
      if (node.input_size() != 2) {
        return errors::InvalidArgument("RefSwitch node", node.name(),
                                       " does not have exactly 2 inputs.");
      }
      ref_input_name = node.mutable_input(0);
    } else {
      // For other ops, check if their inputs are the ref ops we want to
      // eliminate, and if so, these ops must not require their inputs to be
      // refs.
      std::string ref_input;
      for (const auto& tensor_name : node.input()) {
        std::string input = grappler::NodeName(tensor_name);
        if (ref_nodes.contains(input)) {
          ref_input = std::move(input);
          break;
        }
      }
      if (!ref_input.empty()) {
        const OpDef* op_def;
        TF_RETURN_IF_ERROR(op_factory->LookUpOpDef(node.op(), &op_def));
        // TODO(tfrt-devs): How to match input_args to input names in NodeDef?
        for (const auto& input_arg : op_def->input_arg()) {
          if (input_arg.is_ref()) {
            return errors::Unimplemented(
                "Cannot in-place update ref node ", ref_input,
                " to the non-ref counterpart since its user node ", node.name(),
                " requires its input to be refs.");
          }
        }
      }
    }

    if (ref_input_name != nullptr) {
      std::string identity_name =
          absl::StrCat(grappler::NodeName(*ref_input_name), "/identity");
      if (!new_identities.contains(identity_name)) {
        *updated_graph_def.add_node() =
            CreateNewIdentityNode(node, *ref_input_name, identity_name);
        new_identities.insert(identity_name);
      }
      *ref_input_name = std::move(identity_name);
    }

    *updated_graph_def.add_node() = std::move(node);
  }

  graph_def.mutable_node()->Swap(updated_graph_def.mutable_node());
  return OkStatus();
}

void RemoveInputShapesInFunctions(tensorflow::GraphDef& graph_def) {
  for (tensorflow::FunctionDef& function_def :
       *graph_def.mutable_library()->mutable_function()) {
    function_def.mutable_attr()->erase("_input_shapes");
  }
}

namespace {

// Optimizes the functions in `flib_proto` (filtering with
// `functions_to_optimize`) using `flib` and `fallback_state`. Each
// function is converted to a graph and optimized with Placer and Grappler, then
// converted back to a function to replace the old one.
Status OptimizeFunctions(
    FunctionDefLibrary& flib_proto, const FunctionLibraryDefinition& flib,
    const FallbackState& fallback_state,
    const absl::flat_hash_set<std::string>& functions_to_optimize) {
  for (FunctionDef& fdef : *flib_proto.mutable_function()) {
    if (!functions_to_optimize.contains(fdef.signature().name())) {
      continue;
    }

    // Convert function to graph.
    std::unique_ptr<FunctionBody> fbody;
    TF_RETURN_IF_ERROR(
        FunctionDefToBodyHelper(fdef, AttrSlice(), &flib, &fbody));

    tensorflow::Graph* graph = fbody->graph;
    tensorflow::GraphDef graph_def;
    graph->ToGraphDef(&graph_def);
    // We need to manually add the flib because it's not added in
    // `FunctionDefToBodyHelper()`.
    *graph_def.mutable_library() = flib.ToProto();

    // `CreateGraphExecutionState()` will preprocess the graph (e.g., apply
    // Placer).
    TF_ASSIGN_OR_RETURN(
        auto graph_execution_state,
        fallback_state.CreateGraphExecutionState(std::move(graph_def)));

    // Invoke Grappler to optimize the graph.
    std::unique_ptr<tensorflow::Graph> optimized_graph;
    std::unique_ptr<tensorflow::FunctionLibraryDefinition> optimized_flib;
    tensorflow::BuildGraphOptions build_graph_options;
    std::vector<std::string> args;
    args.reserve(fbody->arg_nodes.size());
    for (const auto& arg : fbody->arg_nodes) args.push_back(arg->name());
    std::vector<std::string> rets;
    rets.reserve(fbody->ret_nodes.size());
    for (const auto& ret : fbody->ret_nodes) rets.push_back(ret->name());
    std::vector<std::string> control_rets;
    control_rets.reserve(fbody->control_ret_nodes.size());
    for (const auto& control_ret : fbody->control_ret_nodes) {
      control_rets.push_back(control_ret->name());
    }
    PopulateCallableOptions(build_graph_options.callable_options, args, rets,
                            control_rets);
    auto status = graph_execution_state->OptimizeGraph(
        build_graph_options, *graph_execution_state->full_graph(), &flib,
        &optimized_graph, &optimized_flib);

    if (!status.ok()) {
      LOG(ERROR) << "TFRT failed to optimize graph (converted from function: "
                 << fdef.signature().name() << "): " << status;
      continue;
    }

    TF_RETURN_IF_ERROR(
        optimized_graph->AddFunctionLibrary(optimized_flib->ToProto()));

    // Convert graph back to function.
    // We need to store the conversion result into a new `FunctionDef` first to
    // avoid errors.
    FunctionDef new_fdef;
    TF_RETURN_IF_ERROR(GraphToFunctionDef(*optimized_graph,
                                          fdef.signature().name(), &new_fdef));

    fdef = std::move(new_fdef);
  }
  return OkStatus();
}

}  // namespace

StatusOr<std::unique_ptr<tensorflow::Graph>>
TfrtGraphExecutionState::OptimizeGraph(
    const tensorflow::Graph& graph,
    const tensorflow::BuildGraphOptions& build_graph_options) {
  std::unique_ptr<tensorflow::Graph> optimized_graph;
  std::unique_ptr<tensorflow::FunctionLibraryDefinition> optimized_flib;

  {
    absl::MutexLock lock(&graph_execution_state_mu_);
    // Invoke Grappler to optimize the graph.
    TF_RETURN_IF_ERROR(graph_execution_state_->OptimizeGraph(
        build_graph_options, graph, &graph.flib_def(), &optimized_graph,
        &optimized_flib));
  }

  FunctionDefLibrary optimized_flib_proto = optimized_flib->ToProto();
  if (options_.run_placer_grappler_on_functions) {
    TF_RETURN_IF_ERROR(OptimizeFunctions(optimized_flib_proto, *optimized_flib,
                                         fallback_state_,
                                         functions_to_optimize_));
    // Any optimized function is altered but still has the previous name. To
    // avoid errors when adding the optimized flib, we should clear the current
    // flib first.
    optimized_graph->mutable_flib_def()->Clear();
  }

  TF_RETURN_IF_ERROR(optimized_graph->AddFunctionLibrary(optimized_flib_proto));

  return optimized_graph;
}

// TODO(b/239089915): Clean this up after the logic is implemented in TFXLA
// bridge.
Status BuildXlaLaunchOps(Graph* graph) {
  const auto is_xla_launch_node = [](const Node& n) -> StatusOr<bool> {
    if (!n.IsPartitionedCall()) {
      return false;
    }
    bool xla_must_compile = false;
    const bool has_attribute =
        TryGetNodeAttr(n.attrs(), kXlaMustCompileAttr, &xla_must_compile);
    return has_attribute && xla_must_compile;
  };

  const auto get_xla_function_info = [](const Node& launch)
      -> StatusOr<EncapsulateXlaComputationsPass::XlaFunctionInfo> {
    EncapsulateXlaComputationsPass::XlaFunctionInfo result;
    std::vector<DataType> tin_dtypes;
    TF_RETURN_IF_ERROR(GetNodeAttr(launch.def(), "Tin", &tin_dtypes));
    int variable_start_index = 0;
    for (; variable_start_index < tin_dtypes.size(); ++variable_start_index) {
      if (tin_dtypes.at(variable_start_index) == DT_RESOURCE) break;
    }
    result.variable_start_index = variable_start_index;

    NameAttrList func;
    TF_RETURN_IF_ERROR(GetNodeAttr(launch.attrs(), "f", &func));
    result.function_name = func.name();

    return result;
  };

  return EncapsulateXlaComputationsPass::BuildXlaLaunchOps(
      graph, is_xla_launch_node, get_xla_function_info,
      /*add_edges_to_output_of_downstream_nodes=*/false);
}

}  // namespace tfrt_stub
}  // namespace tensorflow
