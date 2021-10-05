#include <ATen/ATen.h>
#include <algorithm>  // std::min
#include <iostream>

#include <benchmark/benchmark.h>

#define QUANTIZE_BENCHMARK(datatype)                                           \
  (benchmark::State& state) {                                                  \
    const size_t numel = static_cast<size_t>(state.range(0));                  \
    int scale = (rand() % 10) + 1;                                             \
    int zero_point = (rand() % 100) + 1;                                       \
    at::Tensor r = at::rand({numel});                                          \
    at::Tensor q;                                                              \
    for (auto _ : state) {                                                     \
      q = at::quantize_per_tensor(r, scale, zero_point, datatype);             \
    }                                                                          \
  }

#define DEQUANTIZE_BENCHMARK(datatype)                                         \
  (benchmark::State& state) {                                                  \
    const size_t numel = static_cast<size_t>(state.range(0));                  \
    int scale = (rand() % 10) + 1;                                             \
    int zero_point = (rand() % 100) + 1;                                       \
    at::Tensor r1 = at::rand({numel});                                         \
    at::Tensor q = at::quantize_per_tensor(r1, scale, zero_point, datatype);   \
    at::Tensor r2;                                                             \
    for (auto _ : state) {                                                     \
      r2 = q.dequantize();                                                     \
    }                                                                          \
  }

static void quantize_quint8 QUANTIZE_BENCHMARK(at::kQUInt8)
static void quantize_qint32 QUANTIZE_BENCHMARK(at::kQInt32)

static void dequantize_quint8 DEQUANTIZE_BENCHMARK(at::kQUInt8)
static void dequantize_qint8 DEQUANTIZE_BENCHMARK(at::kQInt8)

static void dequantize_qint32 DEQUANTIZE_BENCHMARK(at::kQInt32)

static void GenerateNumel(benchmark::internal::Benchmark* b) {
  b->ArgNames({"numel"});
  for (size_t numel = 1; numel <= 1048576 /* 2^20 */; numel *= 2) {
    b->Args({numel});
  }
}

BENCHMARK(quantize_quint8)->Apply(GenerateNumel)->Threads(4)->UseRealTime();
BENCHMARK(quantize_qint32)->Apply(GenerateNumel)->Threads(4)->UseRealTime();
BENCHMARK(dequantize_quint8)->Apply(GenerateNumel)->Threads(4)->UseRealTime();
BENCHMARK(dequantize_qint8)->Apply(GenerateNumel)->Threads(4)->UseRealTime();
BENCHMARK(dequantize_qint32)->Apply(GenerateNumel)->Threads(4)->UseRealTime();
BENCHMARK_MAIN();
