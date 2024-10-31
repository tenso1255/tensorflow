/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "third_party/grpc/include/grpcpp/security/credentials.h"
#include "third_party/grpc/include/grpcpp/security/server_credentials.h"

namespace tensorflow {
namespace rpc {

// Returns LOAS credentials for use when creating a gRPC server inside Google.
std::shared_ptr<::grpc::ServerCredentials> GetDefaultServerCredentials() {
  return ::grpc::InsecureServerCredentials();
}

// Returns LOAS credentials for use when creating a gRPC channel.
std::shared_ptr<::grpc::ChannelCredentials> GetDefaultChannelCredentials() {
  return ::grpc::InsecureChannelCredentials();
}

}  // namespace rpc
}  // namespace tensorflow
