#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

#include <popstd/ActivationMapping.hpp>
#include <popconv/Convolution.hpp>

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {

port::StatusOr<popconv::ConvParams>
GetConvolutionParameters(const HloInstruction* inst) {

  const Shape& input = inst->operand(0)->shape();
  const Shape& kernel = inst->operand(1)->shape();

  const Window& window(inst->window());

  if (ShapeUtil::Rank(input) != 4 || ShapeUtil::Rank(kernel) != 4) {
    return port::Status(
            port::error::FAILED_PRECONDITION,
            port::StrCat("Poplar supports 2D convolution only: ", inst->name()));
  }

  if (window.dimensions().size() != 2) {
    return port::Status(
            port::error::FAILED_PRECONDITION,
            port::StrCat("Invalid window dimension count on ", inst->name()));
  }

  std::string dtype;
  TF_ASSIGN_OR_RETURN(dtype, PoplarDataType(input));

  std::vector<size_t> input_dims = PoplarShapeFromXlaShape(input);
  std::vector<size_t> kernel_dims = PoplarShapeFromXlaShape(kernel);

  const ConvolutionDimensionNumbers& dims(inst->convolution_dimension_numbers());
  unsigned int n_b = input_dims[dims.batch_dimension()];
  unsigned int n_i = input_dims[dims.feature_dimension()];
  unsigned int n_o = kernel_dims[dims.kernel_output_feature_dimension()];
  unsigned int n_y = input_dims[dims.spatial_dimensions(0)];
  unsigned int n_x = input_dims[dims.spatial_dimensions(1)];
  unsigned int f_y = kernel_dims[dims.kernel_spatial_dimensions(0)];
  unsigned int f_x = kernel_dims[dims.kernel_spatial_dimensions(1)];

  unsigned int s_y = window.dimensions(0).stride();
  unsigned int s_x = window.dimensions(1).stride();

  int pl_y = window.dimensions(0).padding_low();
  int pl_x = window.dimensions(1).padding_low();

  int pu_y = window.dimensions(0).padding_high();
  int pu_x = window.dimensions(1).padding_high();

  unsigned int di_y = window.dimensions(0).base_dilation();
  unsigned int di_x = window.dimensions(1).base_dilation();

  unsigned int dw_y = window.dimensions(0).window_dilation();
  unsigned int dw_x = window.dimensions(1).window_dilation();

  popconv::ConvParams params(dtype,
                             {n_b, n_y, n_x, n_i},
                             {f_y, f_x, n_o, n_i},
                             {s_y, s_x},
                             {pl_y, pl_x},
                             {pu_y, pu_x},
                             {di_y, di_x},
                             {0, 0},
                             {0, 0},
                             {dw_y, dw_x});

  return params;
}

port::StatusOr<poplar::Tensor>
ShuffleConvolutionInput(const HloInstruction* inst,
                        const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(inst->convolution_dimension_numbers());

  std::vector<unsigned int> shuffle(4);
  shuffle[d.batch_dimension()] = 0;
  shuffle[d.spatial_dimensions(0)] = 1;
  shuffle[d.spatial_dimensions(1)] = 2;
  shuffle[d.feature_dimension()] = 3;

  return tensor.dimShuffle(shuffle);
}

port::StatusOr<poplar::Tensor>
ShuffleConvolutionWeights(const HloInstruction* inst,
                          const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(inst->convolution_dimension_numbers());

  std::vector<unsigned int> shuffle(4);
  shuffle[d.kernel_spatial_dimensions(0)] = 0;
  shuffle[d.kernel_spatial_dimensions(1)] = 1;
  shuffle[d.kernel_output_feature_dimension()] = 2;
  shuffle[d.kernel_input_feature_dimension()] = 3;

  return tensor.dimShuffle(shuffle);
}

port::StatusOr <poplar::program::Program>
CreateConv2D(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape &output_shape,
             TensorMap &tensor_map) {

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0, 0));

  // Find the input tensor
  poplar::Tensor kernel;
  TF_ASSIGN_OR_RETURN(kernel, FindInstructionInput(tensor_map, inst, 1, 0));

  popconv::ConvOptions opts;
  opts.cache = &res.convolution_cache;

  const ConvolutionDimensionNumbers& d(inst->convolution_dimension_numbers());

  popconv::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(inst));

  poplar::program::Sequence prog;

  in = in.dimShuffle({
    (unsigned int)d.batch_dimension(),
    (unsigned int)d.spatial_dimensions(0),
    (unsigned int)d.spatial_dimensions(1),
    (unsigned int)d.feature_dimension()
  });
  poplar::Tensor conv_in = popconv::createInput(graph, params, "", opts);
  prog.add(poplar::program::Copy(in, conv_in));

  kernel = kernel.dimShuffle({
    (unsigned int)d.kernel_spatial_dimensions(0),
    (unsigned int)d.kernel_spatial_dimensions(1),
    (unsigned int)d.kernel_output_feature_dimension(),
    (unsigned int)d.kernel_input_feature_dimension()
  });
  poplar::Tensor conv_kernel = popconv::createWeights(graph, params, "", opts);
  prog.add(poplar::program::Copy(kernel, conv_kernel));

  // TODO If the weight input and output channels are reversed, then we can use
  // TODO the poplar feature the reorder them internally. - this would require
  // TODO the reverse op to be fused with the conv op in the backward pass.

  // Add the convolution
  poplar::Tensor out = popconv::convolution(graph, conv_in, conv_kernel, params,
                                            false, prog, "", opts);

  std::vector<unsigned int> out_shuffle(4);
  out_shuffle[d.batch_dimension()] = 0;
  out_shuffle[d.spatial_dimensions(0)] = 1;
  out_shuffle[d.spatial_dimensions(1)] = 2;
  out_shuffle[d.feature_dimension()] = 3;

  out = out.dimShuffle(out_shuffle);

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

}
}
