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

// XLA-specific Fill Op.

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace {

class FillOp : public XlaOpKernel {
 public:
  explicit FillOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // The output of this Op is a tensor of shape 'dims_shape' with each
    // element set to the scalar 'dims_literal'.
    const TensorShape dims_shape = ctx->InputShape("dims");
    const TensorShape value_shape = ctx->InputShape("value");
    OP_REQUIRES(
        ctx, IsLegacyVector(dims_shape),
        errors::InvalidArgument("dims must be a vector of int32, got shape ",
                                dims_shape.DebugString()));
    OP_REQUIRES(ctx, IsLegacyScalar(value_shape),
                errors::InvalidArgument("value must be a scalar, got shape ",
                                        value_shape.DebugString()));

    // Evaluate the 'dims' constant input, reshaping to a vector if it
    // was a 'legacy' vector (secretly a scalar).
    std::vector<int64> dims;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputReshapedToIntVector("dims", &dims));

    // Look up the value input, reshaping to a scalar if it was a
    // 'legacy' scalar (secretly a vector).
    xla::XlaOp data = ctx->Input("value");
    if (value_shape.dims() > 0) {
      CHECK_EQ(value_shape.dims(), 1);
      data = xla::Reshape(data, {});
    }
    // Emit the actual computation, which broadcasts the scalar to the
    // desired shape.
    auto result = xla::Broadcast(data, dims);

    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("Fill").CompileTimeConstantInput("dims"), FillOp);

}  // namespace
}  // namespace tensorflow
