/* Copyright 2024 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/numa.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {

class GpuExecutorTest : public testing::Test {
 public:
  Platform* GetPlatform() {
    auto name = absl::AsciiStrToLower(
        xla::PlatformUtil::CanonicalPlatformName("gpu").value());
    return PlatformManager::PlatformWithName(name).value();
  }
};

using GetPointerMemorySpaceTest = GpuExecutorTest;

TEST_F(GetPointerMemorySpaceTest, Host) {
  StreamExecutor* executor = GetPlatform()->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto host_ptr, executor->HostMemoryAllocate(64));
  TF_ASSERT_OK_AND_ASSIGN(auto memory_space,
                          executor->GetPointerMemorySpace(host_ptr->opaque()))
  EXPECT_EQ(memory_space, MemoryType::kHost);
}

TEST_F(GetPointerMemorySpaceTest, Device) {
  StreamExecutor* executor = GetPlatform()->ExecutorForDevice(0).value();
  auto mem = executor->Allocate(64);
  ASSERT_NE(mem, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(auto memory_space,
                          executor->GetPointerMemorySpace(mem.opaque()))
  EXPECT_EQ(memory_space, MemoryType::kDevice);
  executor->Deallocate(&mem);
}

using HostMemoryAllocateTest = GpuExecutorTest;

TEST_F(HostMemoryAllocateTest, Numa) {
  Platform* platform = GetPlatform();
  const uint64_t kSize = 1024;
  const int num_devices = platform->VisibleDeviceCount();
  for (int device = 0; device < num_devices; ++device) {
    TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                            platform->ExecutorForDevice(device));
    ASSERT_TRUE(executor);
    TF_ASSERT_OK_AND_ASSIGN(auto device_desc,
                            executor->CreateDeviceDescription());
    ASSERT_TRUE(device_desc);
    TF_ASSERT_OK_AND_ASSIGN(auto host_ptr, executor->HostMemoryAllocate(kSize));
    ASSERT_TRUE(host_ptr);
    EXPECT_NE(host_ptr->opaque(), nullptr);
    const int numa_node = tsl::port::NUMAGetMemAffinity(host_ptr->opaque());
    if (numa_node == tsl::port::kNUMANoAffinity) {
      // Could be because `executor` could not determine its own NUMA node, in
      // which case numa_node() will be -1 or 0, depending on the failure mode.
      EXPECT_LE(device_desc->numa_node(), 0);
      EXPECT_GE(device_desc->numa_node(), -1);
    } else {
      EXPECT_EQ(device_desc->numa_node(), numa_node);
    }
  }
}

}  // namespace stream_executor
