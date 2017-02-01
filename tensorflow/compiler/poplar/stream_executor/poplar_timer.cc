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

#include "tensorflow/compiler/poplar/stream_executor/poplar_timer.h"

#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace gpu = ::perftools::gputools;

namespace perftools {
namespace gputools {
namespace poplar {

using std::chrono::duration_cast;

bool PoplarTimer::Start(Stream* stream) {
  return stream->ThenDoHostCallback([this]() { this->StartNow(); }).ok();
}

bool PoplarTimer::Stop(Stream* stream) {
  return stream->ThenDoHostCallback([this]() { this->StopNow(); }).ok();
}

uint64 PoplarTimer::Microseconds() const {
  return duration_cast<std::chrono::microseconds>(duration_).count();
}

uint64 PoplarTimer::Nanoseconds() const {
  return duration_cast<std::chrono::nanoseconds>(duration_).count();
}

void PoplarTimer::StartNow() { start_time_ = clock::now(); }

void PoplarTimer::StopNow() { duration_ = clock::now() - start_time_; }

}  // namespace poplar
}  // namespace gputools
}  // namespace perftools
