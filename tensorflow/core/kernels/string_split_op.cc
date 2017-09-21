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

// See docs in ../ops/string_ops.cc.

#include <string>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

namespace {

Status Split(const string& str, const string& delimiter, const bool skip_empty,
             const string& encoding, std::vector<string>* result) {
  if (encoding == "" && !delimiter.empty()) {
    if (skip_empty) {
      *result = str_util::Split(str, delimiter, str_util::SkipEmpty());
      return Status::OK();
    }
    *result = str_util::Split(str, delimiter);
    return Status::OK();
  }
  if (encoding == "utf8") {
    // We should verify that delimiter is UTF8 encoded (or an empty string "")
    // here. However, as only one character is allowed in the ops, we just
    // check to make sure the char is `0xxxxxxx`.
    // TODO (yongtang): Should we modify the ops or create a new ops to allow
    // multiple chars for UTF8 encoded strings?
    if ((delimiter[0] & 0x80) != 0x00) {
      return errors::InvalidArgument("Delimiter is not properly encoded");
    }
    TF_RETURN_IF_ERROR(str_util::SplitUTF8(str, delimiter, result));
  } else {
    result->resize(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
      (*result)[i] = str[i];
    }
  }
  return Status::OK();
}

}  // namespace

class StringSplitOp : public OpKernel {
 public:
  explicit StringSplitOp(OpKernelConstruction* context)
      : OpKernel(context), skip_empty_(true) {
    bool skip_empty;
    // By default skip_empty_ is true. We only get the value from attr if it is
    // available, so that it is backward compatible.
    if (context->GetAttr("skip_empty", &skip_empty).ok()) {
      skip_empty_ = skip_empty;
    }
    string encoding;
    if (context->GetAttr("encoding", &encoding).ok()) {
      // If encoding is specified, make sure it is either "" or "utf8".
      OP_REQUIRES(
          context, encoding == "" || encoding == "utf8",
          errors::InvalidArgument("encoding must be either '' or 'utf8'"));
      encoding_ = encoding;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    const Tensor* delimiter_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("delimiter", &delimiter_tensor));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(delimiter_tensor->shape()),
        errors::InvalidArgument("delimiter must be a scalar, got shape: ",
                                delimiter_tensor->shape().DebugString()));
    const auto delimiter_vec = delimiter_tensor->flat<string>();
    const string& delimiter = delimiter_vec(0);
    std::vector<string> tokens;
    // Guess that we'll be unpacking a handful of tokens per example.
    static constexpr int kReserveSize = 4;
    tokens.reserve(batch_size * kReserveSize);

    int64 output_size = 0;
    int64 max_num_entries = 0;
    std::vector<int64> num_indices(batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      std::vector<string> parts;
      OP_REQUIRES_OK(
          ctx, Split(input_vec(i), delimiter, skip_empty_, encoding_, &parts));
      int64 n_entries = parts.size();
      num_indices[i] = n_entries;
      output_size += n_entries;
      max_num_entries = std::max(max_num_entries, n_entries);
      tokens.insert(tokens.end(), parts.begin(), parts.end());
    }

    Tensor* sp_indices_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({output_size, 2}),
                                             &sp_indices_t));
    Tensor* sp_tokens_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({output_size}), &sp_tokens_t));
    Tensor* sp_shape_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({2}), &sp_shape_t));

    auto sp_indices = sp_indices_t->matrix<int64>();
    auto sp_tokens = sp_tokens_t->vec<string>();
    auto sp_shape = sp_shape_t->vec<int64>();
    sp_shape(0) = batch_size;
    sp_shape(1) = max_num_entries;
    size_t c = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      for (size_t j = 0; j < num_indices[i]; ++j) {
        sp_indices(c, 0) = i;
        sp_indices(c, 1) = j;
        sp_tokens(c) = tokens[c];
        ++c;
      }
    }
  }

 private:
  bool skip_empty_;
  string encoding_;
};

REGISTER_KERNEL_BUILDER(Name("StringSplit").Device(DEVICE_CPU), StringSplitOp);

}  // namespace tensorflow
