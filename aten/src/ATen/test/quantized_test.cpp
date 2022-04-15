#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/test/test_assert.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>
// For quantize_val
#include <ATen/native/quantized/affine_quantizer.h>
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>
#include <ATen/quantized/Quantizer.h>

using namespace at;
#ifndef ATEN_CPU_STATIC_DISPATCH

#if defined(__ARM_NEON__) || defined(__aarch64__)
TEST(TestQTensor, TestArmVectorizedQuantizeDequantize) {
  const float scale = 7;
  const int numel = 132;

  std::vector<float> x_values;
  for (const auto i : c10::irange(numel)) {
    x_values.push_back(9 * i);
  }

  const Tensor x = from_blob(x_values.data(), x_values.size());

  auto test_for_datatype = [&](
      const ScalarType scalar_type,
      const auto get_data_ptr,
      const auto quantize_val_with_datatype,
      const int zero_point_min,
      const int zero_point_max) {
    for (int zero_point : {zero_point_min, 10, zero_point_max}) {
      const Tensor q = at::quantize_per_tensor(x, scale, zero_point, scalar_type);
      auto* q_data = get_data_ptr(q);
      bool success = true;
      for (const auto i : c10::irange(numel)) {
        auto v1 = q_data[i].val_;
        auto v2 = quantize_val_with_datatype(scale, zero_point, x_values[i]).val_;
        if (v1 != v2) {
          std::cout << "------------------------------------------------" << std::endl;
          std::cout << "Failed to quantize: x[" << i << "] = " << x_values[i] << "; Got: (" << (int)(v1) << "), Expected: (" << (int)(v2) << ")" << std::endl;
          std::cout << "------------------------------------------------" << std::endl;
          success = false;
        }
      }
      ASSERT_EQ(success, true);
    }
  };
  // Signed Int 8
  test_for_datatype(
    kQInt8,
    [](Tensor q) { return q.data_ptr<qint8>(); },
    native::quantize_val<qint8>,
    std::numeric_limits<int8_t>::min(),
    std::numeric_limits<int8_t>::max());
}
#endif // (__ARM_NEON__) || defined(__aarch64__)

#endif // ATEN_CPU_STATIC_DISPATCH
