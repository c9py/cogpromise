/// Copyright (C) 2022 Electronic Arts Inc.  All rights reserved.
/// Tensor Logic Implementation
/// This implements tensor logic concepts for bridging neural and symbolic AI

#ifndef MINICOROS_TENSOR_LOGIC_H_
#define MINICOROS_TENSOR_LOGIC_H_

#ifdef MINICOROS_CUSTOM_INCLUDE
  #include MINICOROS_CUSTOM_INCLUDE
#endif

#include <minicoros/future.h>
#include <minicoros/types.h>

#ifdef MINICOROS_USE_EASTL
  #include <eastl/vector.h>
  #include <eastl/unordered_map.h>
  #include <eastl/string.h>
  #include <eastl/functional.h>
  #ifndef MINICOROS_STD
    #define MINICOROS_STD eastl
  #endif
#else
  #include <vector>
  #include <unordered_map>
  #include <string>
  #include <functional>
  #include <cmath>
  #include <algorithm>
  #include <numeric>
  #ifndef MINICOROS_STD
    #define MINICOROS_STD std
  #endif
#endif

namespace mc {
namespace tensor_logic {

/// Basic tensor type for representing multi-dimensional data
template<typename T>
class tensor {
public:
  tensor() = default;
  
  /// Create a tensor with given dimensions
  explicit tensor(const MINICOROS_STD::vector<size_t>& dims) 
    : dimensions_(dims) {
    size_t total_size = 1;
    for (auto dim : dims) {
      total_size *= dim;
    }
    data_.resize(total_size);
  }
  
  /// Create a tensor from existing data
  tensor(const MINICOROS_STD::vector<size_t>& dims, const MINICOROS_STD::vector<T>& data)
    : dimensions_(dims), data_(data) {}
  
  /// Get dimensions
  const MINICOROS_STD::vector<size_t>& dimensions() const { return dimensions_; }
  
  /// Get total number of elements
  size_t size() const { return data_.size(); }
  
  /// Access element by flat index
  T& operator[](size_t idx) { return data_[idx]; }
  const T& operator[](size_t idx) const { return data_[idx]; }
  
  /// Access element by multi-dimensional index
  T& at(const MINICOROS_STD::vector<size_t>& indices) {
    return data_[flatten_index(indices)];
  }
  
  const T& at(const MINICOROS_STD::vector<size_t>& indices) const {
    return data_[flatten_index(indices)];
  }
  
  /// Get raw data
  MINICOROS_STD::vector<T>& data() { return data_; }
  const MINICOROS_STD::vector<T>& data() const { return data_; }
  
  /// Fill tensor with a value
  void fill(T value) {
    for (auto& elem : data_) {
      elem = value;
    }
  }
  
private:
  size_t flatten_index(const MINICOROS_STD::vector<size_t>& indices) const {
    size_t flat_idx = 0;
    size_t stride = 1;
    for (int i = dimensions_.size() - 1; i >= 0; --i) {
      flat_idx += indices[i] * stride;
      stride *= dimensions_[i];
    }
    return flat_idx;
  }
  
  MINICOROS_STD::vector<size_t> dimensions_;
  MINICOROS_STD::vector<T> data_;
};

/// Predicate represents a logical relation in tensor form
template<typename T>
class predicate {
public:
  predicate() = default;
  predicate(const MINICOROS_STD::string& name, const tensor<T>& tensor_repr)
    : name_(name), tensor_repr_(tensor_repr) {}
  
  const MINICOROS_STD::string& name() const { return name_; }
  const tensor<T>& tensor_representation() const { return tensor_repr_; }
  tensor<T>& tensor_representation() { return tensor_repr_; }
  
private:
  MINICOROS_STD::string name_;
  tensor<T> tensor_repr_;
};

/// Knowledge base for storing predicates and rules
template<typename T>
class knowledge_base {
public:
  /// Add a predicate to the knowledge base
  void add_predicate(const MINICOROS_STD::string& name, const tensor<T>& tensor_repr) {
    predicates_[name] = predicate<T>(name, tensor_repr);
  }
  
  /// Get a predicate by name
  predicate<T>* get_predicate(const MINICOROS_STD::string& name) {
    auto it = predicates_.find(name);
    if (it != predicates_.end()) {
      return &it->second;
    }
    return nullptr;
  }
  
  /// Check if predicate exists
  bool has_predicate(const MINICOROS_STD::string& name) const {
    return predicates_.find(name) != predicates_.end();
  }
  
  /// Get all predicate names
  MINICOROS_STD::vector<MINICOROS_STD::string> get_predicate_names() const {
    MINICOROS_STD::vector<MINICOROS_STD::string> names;
    for (const auto& pair : predicates_) {
      names.push_back(pair.first);
    }
    return names;
  }
  
private:
  MINICOROS_STD::unordered_map<MINICOROS_STD::string, predicate<T>> predicates_;
};

/// Tensor operations for logical inference
namespace operations {

/// Matrix multiplication for 2D tensors (basic join operation)
template<typename T>
tensor<T> matmul(const tensor<T>& a, const tensor<T>& b) {
  const auto& a_dims = a.dimensions();
  const auto& b_dims = b.dimensions();
  
  // For 2D tensors: (m, n) x (n, p) = (m, p)
  if (a_dims.size() != 2 || b_dims.size() != 2 || a_dims[1] != b_dims[0]) {
    return tensor<T>();
  }
  
  size_t m = a_dims[0];
  size_t n = a_dims[1];
  size_t p = b_dims[1];
  
  tensor<T> result({m, p});
  result.fill(T{});
  
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      T sum = T{};
      for (size_t k = 0; k < n; ++k) {
        sum += a[i * n + k] * b[k * p + j];
      }
      result[i * p + j] = sum;
    }
  }
  
  return result;
}

/// Element-wise multiplication
template<typename T>
tensor<T> hadamard(const tensor<T>& a, const tensor<T>& b) {
  if (a.dimensions() != b.dimensions()) {
    return tensor<T>();
  }
  
  tensor<T> result(a.dimensions());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] * b[i];
  }
  
  return result;
}

/// Element-wise addition
template<typename T>
tensor<T> add(const tensor<T>& a, const tensor<T>& b) {
  if (a.dimensions() != b.dimensions()) {
    return tensor<T>();
  }
  
  tensor<T> result(a.dimensions());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  
  return result;
}

/// Apply softmax for probabilistic reasoning
template<typename T>
tensor<T> softmax(const tensor<T>& input, T temperature = T(1.0)) {
  tensor<T> result(input.dimensions());
  
  // Find max for numerical stability
  T max_val = input[0];
  for (size_t i = 1; i < input.size(); ++i) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }
  
  // Compute exp(x/temperature) and sum
  T sum = T{};
  for (size_t i = 0; i < input.size(); ++i) {
    T exp_val = MINICOROS_STD::exp((input[i] - max_val) / temperature);
    result[i] = exp_val;
    sum += exp_val;
  }
  
  // Normalize
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] /= sum;
  }
  
  return result;
}

/// Transpose a 2D tensor
template<typename T>
tensor<T> transpose(const tensor<T>& input) {
  const auto& dims = input.dimensions();
  if (dims.size() != 2) {
    return tensor<T>();
  }
  
  size_t m = dims[0];
  size_t n = dims[1];
  
  tensor<T> result({n, m});
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      result[j * m + i] = input[i * n + j];
    }
  }
  
  return result;
}

} // namespace operations

/// Reasoning engine that supports both symbolic and neural-style inference
template<typename T>
class reasoning_engine {
public:
  explicit reasoning_engine(T temperature = T(1.0)) 
    : temperature_(temperature) {}
  
  /// Set reasoning temperature (0 = deterministic, >0 = increasingly probabilistic)
  void set_temperature(T temp) { temperature_ = temp; }
  T get_temperature() const { return temperature_; }
  
  /// Perform logical join (equivalent to relational join in databases)
  /// This is a core operation in tensor logic
  tensor<T> logical_join(const tensor<T>& predicate1, const tensor<T>& predicate2) {
    return operations::matmul(predicate1, predicate2);
  }
  
  /// Perform logical composition with temperature-based reasoning
  tensor<T> compose_with_temperature(const tensor<T>& predicate1, 
                                     const tensor<T>& predicate2) {
    auto joined = logical_join(predicate1, predicate2);
    if (temperature_ > T(0)) {
      return operations::softmax(joined, temperature_);
    }
    return joined;
  }
  
  /// Query the knowledge base with a pattern
  mc::future<tensor<T>> query(knowledge_base<T>& kb, 
                               const MINICOROS_STD::string& predicate_name) {
    return mc::future<tensor<T>>([&kb, predicate_name](mc::promise<tensor<T>> promise) {
      auto* pred = kb.get_predicate(predicate_name);
      if (pred) {
        tensor<T> result = pred->tensor_representation();
        promise(mc::concrete_result<tensor<T>>(MINICOROS_STD::move(result)));
      } else {
        promise(mc::concrete_result<tensor<T>>(mc::failure(1))); // Predicate not found
      }
    });
  }
  
  /// Infer new knowledge by composing existing predicates
  mc::future<tensor<T>> infer(knowledge_base<T>& kb,
                              const MINICOROS_STD::string& pred1_name,
                              const MINICOROS_STD::string& pred2_name) {
    return mc::future<tensor<T>>([this, &kb, pred1_name, pred2_name]
                                  (mc::promise<tensor<T>> promise) {
      auto* pred1 = kb.get_predicate(pred1_name);
      auto* pred2 = kb.get_predicate(pred2_name);
      
      if (pred1 && pred2) {
        tensor<T> result = compose_with_temperature(pred1->tensor_representation(),
                                                    pred2->tensor_representation());
        promise(mc::concrete_result<tensor<T>>(MINICOROS_STD::move(result)));
      } else {
        promise(mc::concrete_result<tensor<T>>(mc::failure(1))); // Predicate not found
      }
    });
  }
  
private:
  T temperature_;
};

/// Embedding space for neural-symbolic integration
template<typename T>
class embedding_space {
public:
  /// Create an embedding space with specified dimension
  explicit embedding_space(size_t dimension) : dimension_(dimension) {}
  
  /// Add an entity with its embedding vector
  void add_entity(const MINICOROS_STD::string& entity_name, 
                  const MINICOROS_STD::vector<T>& embedding) {
    if (embedding.size() == dimension_) {
      embeddings_[entity_name] = embedding;
    }
  }
  
  /// Get embedding for an entity
  const MINICOROS_STD::vector<T>* get_embedding(const MINICOROS_STD::string& entity_name) const {
    auto it = embeddings_.find(entity_name);
    if (it != embeddings_.end()) {
      return &it->second;
    }
    return nullptr;
  }
  
  /// Compute similarity between two entities (cosine similarity)
  T similarity(const MINICOROS_STD::string& entity1, 
               const MINICOROS_STD::string& entity2) const {
    auto* emb1 = get_embedding(entity1);
    auto* emb2 = get_embedding(entity2);
    
    if (!emb1 || !emb2) {
      return T{};
    }
    
    T dot_product = T{};
    T norm1 = T{};
    T norm2 = T{};
    
    for (size_t i = 0; i < dimension_; ++i) {
      dot_product += (*emb1)[i] * (*emb2)[i];
      norm1 += (*emb1)[i] * (*emb1)[i];
      norm2 += (*emb2)[i] * (*emb2)[i];
    }
    
    if (norm1 > T{} && norm2 > T{}) {
      return dot_product / (MINICOROS_STD::sqrt(norm1) * MINICOROS_STD::sqrt(norm2));
    }
    
    return T{};
  }
  
  size_t dimension() const { return dimension_; }
  
private:
  size_t dimension_;
  MINICOROS_STD::unordered_map<MINICOROS_STD::string, MINICOROS_STD::vector<T>> embeddings_;
};

} // namespace tensor_logic
} // namespace mc

#endif // MINICOROS_TENSOR_LOGIC_H_
