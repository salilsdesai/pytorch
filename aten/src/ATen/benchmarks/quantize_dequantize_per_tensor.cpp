#include <ATen/ATen.h>
#include <iostream>

#include <benchmark/benchmark.h>

static void quantize_quint8(benchmark::State& state) {
  const size_t numel = static_cast<size_t>(state.range(0));
  int scale = (rand() % 10) + 1;
  int zero_point = (rand() % 100) + 1;

  at::Tensor r = at::rand({numel});
  at::Tensor q;
  for (auto _ : state) {
    q = at::quantize_per_tensor(r, scale, zero_point, at::kQUInt8);
  }
}

static void quantize_qint32(benchmark::State& state) {
  const size_t numel = static_cast<size_t>(state.range(0));
  int scale = (rand() % 10) + 1;
  int zero_point = (rand() % 100) + 1;

  at::Tensor r = at::rand({numel});
  at::Tensor q;
  for (auto _ : state) {
    q = at::quantize_per_tensor(r, scale, zero_point, at::kQInt32);
  }
}

static void dequantize_quint8(benchmark::State& state) {
  const size_t numel = static_cast<size_t>(state.range(0));
  int scale = (rand() % 10) + 1;
  int zero_point = (rand() % 100) + 1;

  at::Tensor r1 = at::rand({numel});
  at::Tensor q = at::quantize_per_tensor(r1, scale, zero_point, at::kQUInt8);
  at::Tensor r2;
  for (auto _ : state) {
    r2 = q.dequantize();
  }
}

static void dequantize_qint8(benchmark::State& state) {
  const size_t numel = static_cast<size_t>(state.range(0));
  int scale = (rand() % 10) + 1;
  int zero_point = (rand() % 100) + 1;

  at::Tensor r1 = at::rand({numel});
  at::Tensor q = at::quantize_per_tensor(r1, scale, zero_point, at::kQInt8);
  at::Tensor r2;
  for (auto _ : state) {
    r2 = q.dequantize();
  }
}

static void dequantize_qint32(benchmark::State& state) {
  const size_t numel = static_cast<size_t>(state.range(0));
  int scale = (rand() % 10) + 1;
  int zero_point = (rand() % 100) + 1;

  at::Tensor r1 = at::rand({numel});
  at::Tensor q = at::quantize_per_tensor(r1, scale, zero_point, at::kQInt32);
  at::Tensor r2;
  for (auto _ : state) {
    r2 = q.dequantize();
  }
}

static void GenerateNumel(benchmark::internal::Benchmark* b) {
  b->ArgNames({"numel"});
  for (size_t numel = 1; numel <= 4096; numel *= 2) {
    b->Args({numel});
  }
}

BENCHMARK(quantize_quint8)->Apply(GenerateNumel);
BENCHMARK(quantize_qint32)->Apply(GenerateNumel);
BENCHMARK(dequantize_quint8)->Apply(GenerateNumel);
BENCHMARK(dequantize_qint8)->Apply(GenerateNumel);
BENCHMARK(dequantize_qint32)->Apply(GenerateNumel);
BENCHMARK_MAIN();
