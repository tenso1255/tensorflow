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
#include "tensorflow/lite/c/c_api_for_testing.h"

#include "tensorflow/lite/core/c/c_api.h"

extern "C" {

int32_t TfLiteInterpreterOptionsGetNumThreads(
    TfLiteInterpreterOptions* options) {
  return options->num_threads;
}

}  // extern "C"
