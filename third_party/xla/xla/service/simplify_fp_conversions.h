/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_SIMPLIFY_FP_CONVERSIONS_H_
#define XLA_SERVICE_SIMPLIFY_FP_CONVERSIONS_H_

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {

// Simplifies chains of floating-point conversions.
//
// The algebraic simplifier will remove convert pairs of the form `X -> Y -> X`,
// only when they are a no-op (e.g. `bf16 -> f32 -> bf16`). This passes does
// similar, but has two scopes:
// - kSimplifyAllConversions: Simplify any chain of float conversions, possibly
//   improving  accuracy (e.g. `f32 -> bf16 -> f32` is removed).
// - kOnlySimplifyCompilerGeneratedConversions: Only simplify chains of float
//   conversions generated by the compiler in one of the previous optimization
//   passes.
class SimplifyFPConversions : public HloModulePass {
 public:
  enum class Scope {
    kOnlySimplifyCompilerGeneratedConversions,
    kSimplifyAllConversions
  };

  explicit SimplifyFPConversions(Scope scope) : scope_(scope) {}

  absl::string_view name() const override { return "simplify-fp-conversions"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  Scope scope_;
};

}  // namespace xla

#endif  // XLA_SERVICE_SIMPLIFY_FP_CONVERSIONS_H_
