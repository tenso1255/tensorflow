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

#include "xla/service/llvm_ir/llvm_command_line_options.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "llvm/Support/CommandLine.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace llvm_ir {

LLVMCommandLineOptionsLock::LLVMCommandLineOptionsLock(
    const std::vector<std::string>& options)
    : signature_(absl::HashOf(options)) {
  // Wait until other clients are done using LLVM.
  auto no_competing_clients = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    return signature_ == active_signature_ || num_active_clients_ == 0;
  };
  lock_.LockWhen(absl::Condition(&no_competing_clients));

  // Check if previous client used a different set of LLVM options,
  // (re)initialize if that's the case.
  if (signature_ != active_signature_) {
    LOG(INFO) << "XLA (re)initializing LLVM with options fingerprint: "
              << signature_;
    VLOG(1) << "XLA LLVM options:";
    CHECK_EQ(num_active_clients_, 0);

    int32_t idx = 1;
    std::vector<const char*> fake_argv(options.size() + 1);
    fake_argv[0] = "xla";
    for (std::string_view option : options) {
      VLOG(1) << absl::StrFormat("arg[%d]: %s", idx, option);
      fake_argv[idx] = option.data();
      ++idx;
    }
    // Don't reset LLVM on initialization, as clients might have previously
    // consumed real CLI args.
    if (active_signature_ != 0) {
      llvm::cl::ResetAllOptionOccurrences();
    }
    llvm::cl::ParseCommandLineOptions(fake_argv.size(), fake_argv.data());
    active_signature_ = signature_;
  } else {
    VLOG(1) << "XLA skipping reinitializing LLVM with options signature: "
            << signature_;
  }

  // We're good to start compilation.
  num_active_clients_ += 1;
  lock_.Unlock();
}

LLVMCommandLineOptionsLock::~LLVMCommandLineOptionsLock() {
  absl::MutexLock lock(&lock_);
  CHECK_GT(num_active_clients_, 0);
  num_active_clients_ -= 1;
}

}  // namespace llvm_ir
}  // namespace xla
