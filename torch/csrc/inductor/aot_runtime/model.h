#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aot_runtime/proxy_executor.h>

#define AOT_VECTOR_SIZE_CHECK(vec, expected_size) \
  {                                               \
    auto actual_size = vec.size();                \
    TORCH_CHECK(                                  \
        actual_size == expected_size,             \
        "expected vector size to be ",            \
        std::to_string(expected_size),            \
        ", but got ",                             \
        std::to_string(actual_size));             \
  }

namespace torch {
namespace aot_inductor {

// Defines the base class for AOTInductorModel, which is generated by the
// AOTInductor cpp codegen. Since we do not need dynamic dispatch, we rely
// on curiously recurring template pattern (CRTP) to save some runtime
// v-table overhead. The generated AOTInductorModel is specialized with
// methods such as run_impl and members like shape params used for dynamic
// shape cases.
template <typename Model>
class AOTInductorModelBase {
 public:
  AOTInductorModelBase(size_t num_inputs, size_t num_outputs)
      : inputs_info_(num_inputs), outputs_info_(num_outputs) {
    C10_CUDA_CHECK(cudaEventCreate(&run_finished_));
  }

  ~AOTInductorModelBase() {
    C10_CUDA_CHECK(cudaEventDestroy(run_finished_));
  }

  AOTInductorModelBase(AOTInductorModelBase&&) = delete;
  AOTInductorModelBase& operator=(AOTInductorModelBase&&) = delete;
  AOTInductorModelBase(const AOTInductorModelBase&) = delete;
  AOTInductorModelBase& operator=(const AOTInductorModelBase&) = delete;

  // Currently, we assume that constants are passed as a part of the inputs.
  // Passes such as constant-folding may affect how we handle constants.
  // We will revisit it once all the relevant pieces are ready.
  void run(
      const std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      cudaStream_t stream,
      ProxyExecutor* proxy_executor = nullptr) {
    AOT_VECTOR_SIZE_CHECK(inputs, num_inputs());
    AOT_VECTOR_SIZE_CHECK(outputs, num_outputs());

    auto* model = static_cast<Model*>(this);
    model->run_impl(inputs, outputs, stream, proxy_executor);
    C10_CUDA_CHECK(cudaEventRecord(run_finished_, stream));
  }

  size_t num_inputs() const {
    return inputs_info_.size();
  }

  size_t num_outputs() const {
    return outputs_info_.size();
  }

  const char* input_name(int64_t idx) const {
    return inputs_info_.at(idx).name;
  }

  const char* output_name(int64_t idx) const {
    return outputs_info_.at(idx).name;
  }

  const char* get_input_dtype(int64_t idx) const {
    return inputs_info_.at(idx).dtype;
  }

  const char* get_output_dtype(int64_t idx) const {
    return outputs_info_.at(idx).dtype;
  }

  std::vector<int64_t> max_input_shape(int64_t idx) const {
    return max_shape(inputs_info_, idx);
  }

  std::vector<int64_t> max_output_shape(int64_t idx) const {
    return max_shape(outputs_info_, idx);
  }

  /// Returns true if the model is complete.
  bool is_finished() {
    auto event_status = cudaEventQuery(run_finished_);
    if (event_status == cudaSuccess) {
      return true;
    } else if (event_status == cudaErrorNotReady) {
      return false;
    }

    throw std::runtime_error(
        std::string("The model did not finish successfully. Error: ") +
        cudaGetErrorString(cudaGetLastError()));
  }

  /// Synchronizes completion event.
  void wait_for_completion() {
    C10_CUDA_CHECK(cudaEventSynchronize(run_finished_));
  }

 protected:
  class DimInfo {
   public:
    DimInfo(int64_t lb, int64_t ub, int64_t* val_ptr)
        : lower_bound_(lb), upper_bound_(ub), value_ptr_(val_ptr) {}

    void set_value(int64_t val) {
      TORCH_CHECK(
          val < lower_bound_ || val > upper_bound_,
          "dim value out of bounds: expected value to be in [",
          std::to_string(lower_bound_),
          ", ",
          std::to_string(upper_bound_),
          "], but got ",
          std::to_string(val));
      *value_ptr_ = val;
    }

    int64_t lower_bound() const {
      return lower_bound_;
    }

    int64_t upper_bound() const {
      return upper_bound_;
    }

   private:
    int64_t lower_bound_;
    int64_t upper_bound_;
    int64_t* value_ptr_;
  };

  struct ParamInfo {
    const char* name = nullptr;
    const char* dtype = nullptr;
    std::vector<DimInfo> shape;
  };

  std::vector<ParamInfo> inputs_info_;
  std::vector<ParamInfo> outputs_info_;

  // Record if the model finishes an inference run so that its owning
  // AOTModelContainer can re-use this instance.
  cudaEvent_t run_finished_;

 private:
  std::vector<int64_t> max_shape(
      const std::vector<ParamInfo>& params,
      int64_t idx) const {
    std::vector<int64_t> max_shape;
    const ParamInfo& param = params.at(idx);
    auto rank = param.shape.size();
    max_shape.reserve(rank);
    for (size_t i = 0; i < rank; i++) {
      max_shape.push_back(param.shape[i].upper_bound());
    }
    return max_shape;
  }
};

class AOTInductorModel : public AOTInductorModelBase<AOTInductorModel> {
 public:
  AOTInductorModel();

  void run_impl(
      const std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      cudaStream_t stream,
      ProxyExecutor* proxy_executor = nullptr);

  static std::unique_ptr<AOTInductorModel> Create() {
    return std::make_unique<AOTInductorModel>();
  }
};

} // namespace aot_inductor
} // namespace torch
