/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/tensorrt/resources/trt_allocator.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tensorrt {

bool RunTest(const size_t alignment, const size_t size,
             const intptr_t orig_ptr_val, const size_t orig_space) {
  void* const orig_ptr = reinterpret_cast<void*>(orig_ptr_val);
  void* ptr = orig_ptr;
  size_t space = orig_space;
  void* result = Align(alignment, size, ptr, space);
  if (result == nullptr) {
    EXPECT_EQ(orig_ptr, ptr);
    EXPECT_EQ(orig_space, space);
    return false;
  } else {
    EXPECT_EQ(result, ptr);
    const intptr_t ptr_val = reinterpret_cast<intptr_t>(ptr);
    EXPECT_EQ(0, ptr_val % alignment);
    EXPECT_GE(ptr_val, orig_ptr_val);
    EXPECT_GE(space, size);
    EXPECT_LE(space, orig_space);
    EXPECT_EQ(ptr_val + space, orig_ptr_val + orig_space);
    return true;
  }
}

TEST(TRTAllocatorTest, Align) {
  for (const size_t space :
       {1, 2, 3, 4, 7, 8, 9, 10, 16, 32, 511, 512, 513, 700, 12345}) {
    for (size_t alignment = 1; alignment <= space * 4; alignment *= 2) {
      for (const intptr_t ptr_val :
           {1ul, alignment == 1 ? 1ul : alignment - 1, alignment, alignment + 1,
            alignment + (alignment / 2)}) {
        if (ptr_val % alignment == 0) {
          for (const size_t size :
               {1ul, space == 1 ? 1ul : space - 1, space, space + 1}) {
            EXPECT_EQ(space >= size, RunTest(alignment, size, ptr_val, space));
          }
        } else {
          EXPECT_FALSE(RunTest(alignment, space, ptr_val, space));
          const size_t diff = alignment - ptr_val % alignment;
          if (space > diff) {
            EXPECT_TRUE(
                RunTest(alignment, space - diff, ptr_val + diff, space - diff));
            for (const size_t size :
                 {1ul, space - diff > 1 ? space - diff - 1 : 1ul, space - diff,
                  space - diff + 1, space - 1}) {
              EXPECT_EQ(space - diff >= size,
                        RunTest(alignment, size, ptr_val, space));
            }
          } else {
            EXPECT_FALSE(RunTest(alignment, 1, ptr_val, space));
          }
        }
      }
    }
  }
}

}  // namespace tensorrt
}  // namespace tensorflow
