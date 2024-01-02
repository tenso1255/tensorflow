/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SERVING_DEVICE_SELECTOR_POLICIES_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SERVING_DEVICE_SELECTOR_POLICIES_H_

#include <atomic>

#include "tensorflow/core/common_runtime/serving_device_selector.h"

namespace tensorflow {

enum class ServingDeviceSelectorPolicy {
  kRoundRobin,
};

class RoundRobinPolicy : public ServingDeviceSelector::Policy {
 public:
  RoundRobinPolicy() : ordinal_(0) {}

  int SelectDevice(
      absl::string_view program_fingerprint,
      const ServingDeviceSelector::DeviceStates& device_states) override;

 private:
  std::atomic<uint64_t> ordinal_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SERVING_DEVICE_SELECTOR_POLICIES_H_
