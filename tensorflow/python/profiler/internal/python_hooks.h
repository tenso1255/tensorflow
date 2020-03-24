/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
#define TENSORFLOW_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_

#include "absl/container/flat_hash_map.h"
#include "include/pybind11/cast.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace profiler {

namespace py = ::pybind11;

// Singleton for tracing python function calls.
class PythonHooks {
 public:
  static PythonHooks* GetSingleton();

  void Start();
  void Stop();
  void Finalize();
  void ProfileSlow(const py::object& frame, const string& event,
                   const py::object& arg);
  void ProfileFast(PyFrameObject* frame, int what, PyObject* arg);

 private:
  void SetProfilerInAllThreads();
  void ClearProfilerInAllThreads();

  absl::flat_hash_map<int64, std::vector<std::unique_ptr<TraceMe>>> tracemes_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
