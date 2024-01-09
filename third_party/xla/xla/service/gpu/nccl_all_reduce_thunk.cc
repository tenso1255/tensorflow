/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/nccl_all_reduce_thunk.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

#if XLA_ENABLE_XCCL
#include "xla/stream_executor/gpu/gpu_stream.h"
#endif

namespace xla {
namespace gpu {

using mlir::lmhlo_gpu::AllReduceStartOp;
using mlir::lmhlo_gpu::ReduceScatterStartOp;

Status RunAllReduce(ReductionKind reduction_kind,
                    std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                    ncclComm_t comm) {
#if XLA_ENABLE_XCCL
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing all-reduce from device ordinal: " << device_ordinal;

  ncclRedOp_t reduce_op = ToNcclReduction(reduction_kind);

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (size_t i = 0; i < buffers.size(); ++i) {
    DeviceBufferPair& buffer = buffers[i];
    const void* send_buffer = buffer.source_buffer.opaque();
    void* recv_buffer = buffer.destination_buffer.opaque();

    TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                        ToNcclDataTypeAndCountMultiplier(
                            buffer.element_type, Thunk::kNcclAllReduce));
    ncclDataType_t dtype = dtype_and_multiplier.first;
    int64_t element_count = buffer.element_count * dtype_and_multiplier.second;

    VLOG(3) << absl::StreamFormat(
        "Calling ncclAllReduce(send_buffer=%p, recv_buffer=%p, count=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, element_count, static_cast<const void*>(comm),
        gpu_stream);

    XLA_CUDA_RETURN_IF_ERROR(ncclAllReduce(send_buffer, recv_buffer,
                                           element_count, dtype, reduce_op,
                                           comm, gpu_stream));
  }
  return XLA_CUDA_STATUS(ncclGroupEnd());
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

namespace {

// Generally, the reduction op should be the only operation in the block, except
// the terminator. However, if the type is bf16, the `FloatNormalization`
// pass will have converted the op to float32 and added type conversions.
// TODO(cjfj): Can we prevent the bf16 conversion for this computation?
StatusOr<mlir::Operation*> FindReductionOp(mlir::Block& block) {
  TF_RET_CHECK(block.getNumArguments() == 2);
  mlir::Operation* terminator = block.getTerminator();
  TF_RET_CHECK(terminator);
  TF_RET_CHECK(terminator->getNumOperands() == 1);
  mlir::Value result = terminator->getOperand(0);
  TF_RET_CHECK(block.getArgument(0).getType() == result.getType());
  TF_RET_CHECK(block.getArgument(1).getType() == result.getType());

  mlir::Operation* result_op = result.getDefiningOp();
  TF_RET_CHECK(result_op);

  // In the bf16 case, the type conversions and op might be fused.
  if (mlir::isa<mlir::mhlo::FusionOp>(result_op)) {
    return FindReductionOp(result_op->getRegion(0).front());
  }

  // Standard case.
  if (absl::c_is_permutation(result_op->getOperands(), block.getArguments())) {
    return result_op;
  }

  // bf16 case.
  TF_RET_CHECK(mlir::isa<mlir::mhlo::ConvertOp>(result_op));
  TF_RET_CHECK(result_op->getNumOperands() == 1);
  mlir::Operation* reduction_op = result_op->getOperand(0).getDefiningOp();
  TF_RET_CHECK(reduction_op);
  TF_RET_CHECK(reduction_op->getNumOperands() == 2);
  mlir::Value operand0 = reduction_op->getOperand(0);
  mlir::Value operand1 = reduction_op->getOperand(1);
  auto operand0_op = operand0.getDefiningOp<mlir::mhlo::ConvertOp>();
  auto operand1_op = operand1.getDefiningOp<mlir::mhlo::ConvertOp>();
  TF_RET_CHECK(operand0_op);
  TF_RET_CHECK(operand1_op);
  TF_RET_CHECK(operand0_op->getNumOperands() == 1);
  TF_RET_CHECK(operand1_op->getNumOperands() == 1);
  std::array<mlir::Value, 2> operands{operand0_op->getOperand(0),
                                      operand1_op->getOperand(0)};
  TF_RET_CHECK(absl::c_is_permutation(operands, block.getArguments()));
  return reduction_op;
}

}  // namespace

namespace impl {

Status CheckImplementableInst(const HloInstruction* inst,
                              Thunk::Kind reduction_op) {
  TF_RETURN_IF_ERROR(NcclCollectiveThunk::CheckImplementable());

  for (HloInstruction* operand : inst->operands()) {
    TF_RETURN_IF_ERROR(IsValidOperand(operand->shape(), reduction_op));
  }

  if (!MatchReductionComputation(inst->called_computations().front())
           .has_value()) {
    return tsl::errors::Unimplemented("Unrecognized reduction computation");
  }

  return OkStatus();
}

template <typename OpT>
Status CheckImplementable(OpT op, Thunk::Kind reduction_op) {
  TF_RETURN_IF_ERROR(NcclCollectiveThunk::CheckImplementable());
  for (mlir::Value operand : op.getInputs()) {
    TF_RETURN_IF_ERROR(IsValidOperand(operand, reduction_op));
  }
  if (!NcclAllReduceReduceScatterThunkBase::MatchAllReduceComputation(
           op.getComputation())
           .has_value()) {
    return tsl::errors::Unimplemented("Unrecognized reduction computation");
  }
  return OkStatus();
}

template <typename HloInstType>
NcclAllReduceConfig GetNcclAllReduceConfigInst(HloInstType* inst) {
  std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(inst->called_computations().front());
  CHECK(reduction_kind.has_value());

  NcclAllReduceConfig config;
  config.config = GetNcclCollectiveConfig(inst, inst->use_global_device_ids());
  config.reduction_kind = *reduction_kind;
  return config;
}

template <typename OpT>
NcclAllReduceConfig GetNcclAllReduceConfig(OpT op) {
  std::optional<ReductionKind> reduction_kind =
      NcclAllReduceReduceScatterThunkBase::MatchAllReduceComputation(
          op.getComputation());
  CHECK(reduction_kind.has_value());

  NcclAllReduceConfig config;
  config.config =
      GetNcclCollectiveConfigForMlir(op, op.getUseGlobalDeviceIds());
  config.reduction_kind = *reduction_kind;
  return config;
}

template <typename OpT>
CollectiveOpGroupMode GetGroupMode(OpT op) {
  return GetNcclAllReduceConfig(op).config.group_mode;
}

template <typename HloInstType>
CollectiveOpGroupMode GetGroupModeInst(HloInstType* inst) {
  return GetNcclAllReduceConfigInst(inst).config.group_mode;
}

}  // namespace impl

std::optional<ReductionKind>
NcclAllReduceReduceScatterThunkBase::MatchAllReduceComputation(
    mlir::Region& computation) {
  mlir::Block& block = computation.front();
  StatusOr<mlir::Operation*> reduction_op = FindReductionOp(block);
  if (!reduction_op.ok()) return std::nullopt;
  StatusOr<HloOpcode> opcode = MhloToHloOpcode(*reduction_op);
  if (!opcode.ok()) return std::nullopt;
  // Match the operation to a reduction kind. We can represent and/or of pred as
  // min/max. This works because pred is stored as an 8-bit int of value 0 or 1.
  PrimitiveType type =
      TypeToShape(block.getArgument(0).getType()).element_type();
  if (type == PRED) {
    switch (opcode.value()) {
      case HloOpcode::kAnd:
        return ReductionKind::MIN;
      case HloOpcode::kOr:
        return ReductionKind::MAX;
      default:
        return std::nullopt;
    }
  } else if (primitive_util::IsComplexType(type)) {
    // Only addition is supported for complex types.
    if (*opcode == HloOpcode::kAdd) {
      return ReductionKind::SUM;
    } else {
      return std::nullopt;
    }
  } else {
    switch (*opcode) {
      case HloOpcode::kAdd:
        return ReductionKind::SUM;
      case HloOpcode::kMultiply:
        return ReductionKind::PRODUCT;
      case HloOpcode::kMaximum:
        return ReductionKind::MAX;
      case HloOpcode::kMinimum:
        return ReductionKind::MIN;
      default:
        return std::nullopt;
    }
  }
}

NcclAllReduceReduceScatterThunkBase::NcclAllReduceReduceScatterThunkBase(
    Thunk::Kind kind, ThunkInfo thunk_info, NcclAllReduceConfig config,
    std::vector<Buffer> buffers, bool is_sync)
    : NcclCollectiveThunk(kind, thunk_info, is_sync),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

NcclAllReduceStartThunk::NcclAllReduceStartThunk(ThunkInfo thunk_info,
                                                 AllReduceStartOp op,
                                                 std::vector<Buffer> buffers)
    : NcclAllReduceReduceScatterThunkBase(Thunk::kNcclAllReduceStart,
                                          thunk_info,
                                          impl::GetNcclAllReduceConfig(op),
                                          std::move(buffers), op.getIsSync()) {}

NcclAllReduceStartThunk::NcclAllReduceStartThunk(
    ThunkInfo thunk_info, const HloAllReduceInstruction* inst,
    std::vector<Buffer> buffers)
    : NcclAllReduceReduceScatterThunkBase(
          Thunk::kNcclAllReduceStart, thunk_info,
          impl::GetNcclAllReduceConfigInst(inst), std::move(buffers),
          inst->backend_config<CollectiveBackendConfig>()->is_sync()) {}

Status NcclAllReduceStartThunk::CheckImplementable(AllReduceStartOp op,
                                                   int64_t replica_count,
                                                   int64_t partition_count) {
  return AddOpDescription<NcclAllReduceStartThunk>(
      impl::CheckImplementable(op, Thunk::kNcclAllReduceStart), op,
      replica_count, partition_count);
}

Status NcclAllReduceStartThunk::CheckImplementable(
    const HloAllReduceInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<NcclAllReduceStartThunk>(
      impl::CheckImplementableInst(inst, Thunk::kNcclAllReduceStart), inst,
      replica_count, partition_count);
}

CollectiveOpGroupMode NcclAllReduceStartThunk::GetGroupMode(
    AllReduceStartOp op) {
  return impl::GetGroupMode(op);
}

CollectiveOpGroupMode NcclAllReduceStartThunk::GetGroupMode(
    const HloAllReduceInstruction* inst) {
  return impl::GetGroupModeInst(inst);
}

Status NcclAllReduceStartThunk::RunNcclCollective(const ExecuteParams& params,
                                                  se::Stream& stream,
                                                  ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return ::xla::gpu::RunAllReduce(config_.reduction_kind, device_buffers,
                                  stream, comm);
}

NcclReduceScatterStartThunk::NcclReduceScatterStartThunk(
    ThunkInfo thunk_info, ReduceScatterStartOp op,
    std::vector<NcclCollectiveThunk::Buffer> buffers)
    : NcclAllReduceReduceScatterThunkBase(Thunk::kNcclReduceScatterStart,
                                          thunk_info,
                                          impl::GetNcclAllReduceConfig(op),
                                          std::move(buffers), op.getIsSync()) {}

NcclReduceScatterStartThunk::NcclReduceScatterStartThunk(
    ThunkInfo thunk_info, const HloReduceScatterInstruction* inst,
    std::vector<Buffer> buffers)
    : NcclAllReduceReduceScatterThunkBase(
          Thunk::kNcclReduceScatterStart, thunk_info,
          impl::GetNcclAllReduceConfigInst(inst), std::move(buffers),
          inst->backend_config<CollectiveBackendConfig>()->is_sync()) {}

/*static*/ Status NcclReduceScatterStartThunk::CheckImplementable(
    ReduceScatterStartOp op, int64_t replica_count, int64_t partition_count) {
  return AddOpDescription<NcclReduceScatterStartThunk>(
      impl::CheckImplementable(op, Thunk::kNcclReduceScatterStart), op,
      replica_count, partition_count);
}

/*static*/ Status NcclReduceScatterStartThunk::CheckImplementable(
    const HloReduceScatterInstruction* inst, int64_t replica_count,
    int64_t partition_count) {
  return AddOpDescription<NcclReduceScatterStartThunk>(
      impl::CheckImplementableInst(inst, Thunk::kNcclReduceScatterStart), inst,
      replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode NcclReduceScatterStartThunk::GetGroupMode(
    ReduceScatterStartOp op) {
  return impl::GetGroupMode(op);
}

/*static*/ CollectiveOpGroupMode NcclReduceScatterStartThunk::GetGroupMode(
    const HloReduceScatterInstruction* inst) {
  return impl::GetGroupModeInst(inst);
}

Status NcclReduceScatterStartThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream, ncclComm_t comm) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));
  return ::xla::gpu::RunReduceScatter(config_.reduction_kind, device_buffers,
                                      stream, comm);
}

Status RunReduceScatter(ReductionKind reduction_kind,
                        std::vector<DeviceBufferPair>& buffers,
                        se::Stream& stream, ncclComm_t comm) {
#if XLA_ENABLE_XCCL
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing reduce-scatter from device ordinal: "
          << device_ordinal;

  ncclRedOp_t reduce_op = ToNcclReduction(reduction_kind);

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);

  int num_participants = 0;
  XLA_CUDA_RETURN_IF_ERROR(ncclCommCount(comm, &num_participants));

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (size_t i = 0; i < buffers.size(); ++i) {
    DeviceBufferPair& buffer = buffers[i];
    const void* send_buffer = buffer.source_buffer.opaque();
    void* recv_buffer = buffer.destination_buffer.opaque();

    TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                        ToNcclDataTypeAndCountMultiplier(
                            buffer.element_type, Thunk::kNcclReduceScatter));
    ncclDataType_t dtype = dtype_and_multiplier.first;
    int64_t element_count = buffer.element_count * dtype_and_multiplier.second;

    // buffer.element_count is the source buffers element count. For
    // ncclReduceScatter, we need the destination buffers element count.
    TF_RET_CHECK(element_count % num_participants == 0)
        << "Source buffer was not an exact multiple of the number of "
           "participants.";

    int64_t recv_count = element_count / num_participants;
    VLOG(3) << absl::StreamFormat(
        "Calling ncclReduceScatter(send_buffer=%p, recv_buffer=%p, "
        "recvcount=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, recv_count, static_cast<const void*>(comm),
        gpu_stream);
    XLA_CUDA_RETURN_IF_ERROR(ncclReduceScatter(send_buffer, recv_buffer,
                                               recv_count, dtype, reduce_op,
                                               comm, gpu_stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(3) << "Done performing reduce-scatter for ordinal: " << device_ordinal;
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

}  // namespace gpu
}  // namespace xla
