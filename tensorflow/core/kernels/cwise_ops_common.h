/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_CWISE_OPS_COMMON_H_
#define TENSORFLOW_KERNELS_CWISE_OPS_COMMON_H_

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/cwise_ops.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

class BinaryOpShared : public OpKernel {
 public:
  explicit BinaryOpShared(OpKernelConstruction* ctx, DataType out, DataType in);

 protected:
  struct BinaryOpState {
    // Sets up bcast with the shape of in0 and in1, ensures that the bcast
    // is valid, and if so, allocates out using ctx->output(...).
    // Caller must check ctx->status() upon return for non-ok status.
    // If ctx->status().ok() is true, then out is guaranteed to be allocated.
    BinaryOpState(OpKernelContext* ctx);

    BCast bcast;
    Tensor* out = nullptr;
  };

  template <int NDIMS>
  static Eigen::array<Eigen::DenseIndex, NDIMS> ToIndexArray(
      const BCast::Vec& vec) {
    CHECK_EQ(vec.size(), static_cast<size_t>(NDIMS));
    Eigen::array<Eigen::DenseIndex, NDIMS> ret;
    for (int i = 0; i < NDIMS; ++i) ret[i] = vec[i];
    return ret;
  }
  void SetUnimplementedError(OpKernelContext* ctx);
};

// Coefficient-wise binary operations:
//   Device: E.g., CPUDevice, GPUDevice.
//   Functor: defined in cwise_functors.h. E.g., functor::add2.
template <typename Device, typename Functor>
class BinaryOp : public BinaryOpShared {
 public:
  typedef typename Functor::in_type Tin;    // Input scalar data type.
  typedef typename Functor::out_type Tout;  // Output scalar data type.

  explicit BinaryOp(OpKernelConstruction* ctx)
      : BinaryOpShared(ctx, DataTypeToEnum<Tout>::v(),
                       DataTypeToEnum<Tin>::v()) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    // 'state': Shared helper not dependent on T to reduce code size
    BinaryOpState state(ctx);
    if (!ctx->status().ok()) return;
    Tensor* out = state.out;
    BCast* bcast = &state.bcast;
    if (out->NumElements() == 0) {
      return;
    }
    const int ndims = bcast->x_reshape().size();
    if (ndims <= 1) {
      if (in1.NumElements() == 1) {
        // tensor op scalar
        functor::BinaryFunctor<Device, Functor, 1>().Right(
            ctx->eigen_device<Device>(), out->flat<Tout>(), in0.flat<Tin>(),
            in1.scalar<Tin>());
        return;
      }
      if (in0.NumElements() == 1) {
        // scalar op tensor
        functor::BinaryFunctor<Device, Functor, 1>().Left(
            ctx->eigen_device<Device>(), out->flat<Tout>(), in0.scalar<Tin>(),
            in1.flat<Tin>());
        return;
      }
      functor::BinaryFunctor<Device, Functor, 1>()(
          ctx->eigen_device<Device>(), out->flat<Tout>(), in0.flat<Tin>(),
          in1.flat<Tin>());
      return;
    }

    if (ndims == 2) {
      functor::BinaryFunctor<Device, Functor, 2>().BCast(
          ctx->eigen_device<Device>(),
          out->shaped<Tout, 2>(bcast->result_shape()),
          in0.shaped<Tin, 2>(bcast->x_reshape()),
          ToIndexArray<2>(bcast->x_bcast()),
          in1.shaped<Tin, 2>(bcast->y_reshape()),
          ToIndexArray<2>(bcast->y_bcast()));
      return;
    }

    if (ndims == 3) {
      functor::BinaryFunctor<Device, Functor, 3>().BCast(
          ctx->eigen_device<Device>(),
          out->shaped<Tout, 3>(bcast->result_shape()),
          in0.shaped<Tin, 3>(bcast->x_reshape()),
          ToIndexArray<3>(bcast->x_bcast()),
          in1.shaped<Tin, 3>(bcast->y_reshape()),
          ToIndexArray<3>(bcast->y_bcast()));
      return;
    }

    SetUnimplementedError(ctx);
  }

 private:
};

// Coefficient-wise unary operations:
//   Device: E.g., CPUDevice, GPUDevice.
//   Functor: defined in cwise_functors.h. E.g., functor::sqrt.
template <typename Device, typename Functor>
class UnaryOp : public OpKernel {
 public:
  typedef typename Functor::in_type Tin;    // Input scalar data type.
  typedef typename Functor::out_type Tout;  // Output scalar data type.
  // Tin may be different from Tout. E.g., abs: complex64 -> float

  explicit UnaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto in = DataTypeToEnum<Tin>::v();
    auto out = DataTypeToEnum<Tout>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({in}, {out}));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp.shape(), &out));
    functor::UnaryFunctor<Device, Functor>()(
        ctx->eigen_device<Device>(), out->flat<Tout>(), inp.flat<Tin>());
  }
};

// Coefficient-wise select operation.
//   Device: E.g., CPUDevice, GPUDevice.
template <typename Device, typename T>
class SelectOp : public OpKernel {
 public:
  explicit SelectOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({DT_BOOL, dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    const Tensor& in2 = ctx->input(2);
    if (!ctx->ValidateInputsAreSameShape(this)) return;
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in0.shape(), &out));
    functor::SelectFunctor<Device, T> func;
    func(ctx->eigen_device<Device>(), out->flat<T>(), in0.flat<bool>(),
         in1.flat<T>(), in2.flat<T>());
  }
};

namespace functor {

// For CPUDevice, we do operations inline if the resulting tensor is
// modestly sized.
static bool DoInline(size_t size) { return size <= 32768; }

template <typename D, typename OUT, typename RHS>
void Assign(const D& d, OUT out, RHS rhs) {
  if (DoInline(out.size())) {
    out = rhs;
  } else {
    out.device(d) = rhs;
  }
}

// Partial specialization of BinaryFunctor<Device=CPUDevice, Functor>.
template <typename Functor, int NDIMS>
struct BinaryFunctor<CPUDevice, Functor, NDIMS> {
  void operator()(const CPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1) {
    Assign(d, out, in0.binaryExpr(in1, typename Functor::func()));
  }

  void Left(const CPUDevice& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_left<Tout, Tin, Binary> Unary;
    Assign(d, out, in.unaryExpr(Unary(scalar.data())));
  }

  void Right(const CPUDevice& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_right<Tout, Tin, Binary> Unary;
    Assign(d, out, in.unaryExpr(Unary(scalar.data())));
  }

#if !defined(EIGEN_HAS_INDEX_LIST)
  inline Eigen::DSizes<int, 2> NByOne(int n) {
    return Eigen::DSizes<int, 2>(n, 1);
  }
  inline Eigen::DSizes<int, 2> OneByM(int m) {
    return Eigen::DSizes<int, 2>(1, m);
  }
#else
  inline Eigen::IndexList<int, Eigen::type2index<1>> NByOne(int n) {
    Eigen::IndexList<int, Eigen::type2index<1>> ret;
    ret.set(0, n);
    return ret;
  }
  inline Eigen::IndexList<Eigen::type2index<1>, int> OneByM(int m) {
    Eigen::IndexList<Eigen::type2index<1>, int> ret;
    ret.set(1, m);
    return ret;
  }
#endif

  void BCast(const CPUDevice& dev,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1) {
    typedef typename Functor::in_type T;
    typename Functor::func func;
    if ((NDIMS == 2) && Functor::use_bcast_optimization &&
        use_bcast_optimization<T>::value) {
      // Optimize for speed by using Eigen::type2index and avoid
      // .broadcast() when we know its a no-op.
      //
      // Here, we need to handle 6 cases depending on how many "1"
      // exist in in0 and in1's shapes (4 numbers in total). It's not
      // possible that two shapes have more than 2 1s because those
      // are simplified to NDIMS==1 case.
      //
      // Because this optimization increases the binary size for each
      // Functor (+, -, *, /, <, <=, etc.), type and ndim combination.
      // we only apply such optimization for selected ops/types/ndims.
      //
      // Because NDIMS, Functor::use_broadcast_optimization and
      // use_broadcast_optimization<T> are compile-time constant, gcc
      // does a decent job avoiding generating code when conditions
      // are not met.
      const int a = in0.dimension(0);  // in0 is shape [a, b]
      const int b = in0.dimension(1);
      const int c = in1.dimension(0);  // in1 is shape [c, d]
      const int d = in1.dimension(1);
      if ((a == 1) && (d == 1)) {
        auto lhs = in0.reshape(OneByM(b)).broadcast(NByOne(c));
        auto rhs = in1.reshape(NByOne(c)).broadcast(OneByM(b));
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if ((b == 1) && (c == 1)) {
        auto lhs = in0.reshape(NByOne(a)).broadcast(OneByM(d));
        auto rhs = in1.reshape(OneByM(d)).broadcast(NByOne(a));
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if (a == 1) {
        auto lhs = in0.reshape(OneByM(b)).broadcast(NByOne(c));
        auto rhs = in1;
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if (b == 1) {
        auto lhs = in0.reshape(NByOne(a)).broadcast(OneByM(d));
        auto rhs = in1;
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if (c == 1) {
        auto lhs = in0;
        auto rhs = in1.reshape(OneByM(d)).broadcast(NByOne(a));
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if (d == 1) {
        auto lhs = in0;
        auto rhs = in1.reshape(NByOne(c)).broadcast(OneByM(b));
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }

      const bool bcast0_all_one = AllOne<NDIMS>(bcast0);
      const bool bcast1_all_one = AllOne<NDIMS>(bcast1);
      if (bcast0_all_one && !bcast1_all_one) {
        auto lhs = in0;  // No need to do broadcast for in0
        auto rhs = in1.broadcast(bcast1);
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }

      if (!bcast0_all_one && bcast1_all_one) {
        auto lhs = in0.broadcast(bcast0);
        auto rhs = in1;  // No need to do broadcast for in1
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
    }

    // Fallback path. Always work and probably slower.
    auto lhs = in0.broadcast(bcast0);
    auto rhs = in1.broadcast(bcast1);
    Assign(dev, out, lhs.binaryExpr(rhs, func));
  }
};

// Partial specialization of UnaryFunctor<Device=CPUDevice, Functor>.
template <typename Functor>
struct UnaryFunctor<CPUDevice, Functor> {
  void operator()(const CPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in) {
    Assign(d, out, in.unaryExpr(typename Functor::func()));
  }
};

template <typename T>
struct SelectFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
    Assign(d, out, cond_flat.select(then_flat, else_flat));
  }
};

}  // end namespace functor

#define REGISTER_SELECT(D, N, F, T)                                          \
  REGISTER_KERNEL_BUILDER(Name(N).Device(DEVICE_##D).TypeConstraint<T>("T"), \
                          SelectOp<D##Device, T>)

#define REGISTER(OP, D, N, F, T)                                             \
  REGISTER_KERNEL_BUILDER(Name(N).Device(DEVICE_##D).TypeConstraint<T>("T"), \
                          OP<D##Device, F<T>>);

// Macros to register kernels for multiple types (T0, T1, etc.)  on
// device type "D" (CPU or GPU) for operation "N" (e.g., sqrt) using
// the functor "F" (e.g., functor:sqrt).

#if defined(__ANDROID_TYPES_SLIM__)
// Normally Android TensorFlow is built with a reduced number of types (float).
// Override on the command-line "--define ANDROID_TYPES=__ANDROID_TYPES_FULL__"
// to generate a library with full type support with a consequent increase in
// code size.
#define REGISTER2(OP, D, N, F, T0, T1) REGISTER(OP, D, N, F, T0)
#define REGISTER3(OP, D, N, F, T0, T1, T2) REGISTER(OP, D, N, F, T0)
#define REGISTER4(OP, D, N, F, T0, T1, T2, T3) REGISTER(OP, D, N, F, T0)
#define REGISTER5(OP, D, N, F, T0, T1, T2, T3, T4) REGISTER(OP, D, N, F, T0)
#define REGISTER6(OP, D, N, F, T0, T1, T2, T3, T4, T5) REGISTER(OP, D, N, F, T0)
#define REGISTER7(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6) \
  REGISTER(OP, D, N, F, T0)
#define REGISTER8(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6, T7) \
  REGISTER(OP, D, N, F, T0)
#define REGISTER9(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6, T7, T8) \
  REGISTER(OP, D, N, F, T0)
#else  // !defined(__ANDROID_TYPES_SLIM__)
#define REGISTER2(OP, D, N, F, T0, T1) \
  REGISTER(OP, D, N, F, T0)            \
  REGISTER(OP, D, N, F, T1)
#define REGISTER3(OP, D, N, F, T0, T1, T2) \
  REGISTER2(OP, D, N, F, T0, T1)           \
  REGISTER(OP, D, N, F, T2)
#define REGISTER4(OP, D, N, F, T0, T1, T2, T3) \
  REGISTER2(OP, D, N, F, T0, T1)               \
  REGISTER2(OP, D, N, F, T2, T3)
#define REGISTER5(OP, D, N, F, T0, T1, T2, T3, T4) \
  REGISTER3(OP, D, N, F, T0, T1, T2)               \
  REGISTER2(OP, D, N, F, T3, T4)
#define REGISTER6(OP, D, N, F, T0, T1, T2, T3, T4, T5) \
  REGISTER3(OP, D, N, F, T0, T1, T2)                   \
  REGISTER3(OP, D, N, F, T3, T4, T5)
#define REGISTER7(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6) \
  REGISTER4(OP, D, N, F, T0, T1, T2, T3)                   \
  REGISTER3(OP, D, N, F, T4, T5, T6)
#define REGISTER8(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6, T7) \
  REGISTER4(OP, D, N, F, T0, T1, T2, T3)                       \
  REGISTER4(OP, D, N, F, T4, T5, T6, T7)
#define REGISTER9(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6, T7, T8) \
  REGISTER5(OP, D, N, F, T0, T1, T2, T3, T4)                       \
  REGISTER4(OP, D, N, F, T5, T6, T7, T8)
#endif  // defined(__ANDROID_TYPES_SLIM__)

}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CWISE_OPS_COMMON_H_
