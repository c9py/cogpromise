# Tensor Logic Implementation

This document describes the tensor logic implementation in the cogpromise library, which bridges neural and symbolic AI through tensor-based logical reasoning.

## Overview

Tensor Logic is a unified framework that represents both symbolic logical rules and neural network operations as tensor equations. This implementation provides:

- **Tensor data structures** for multi-dimensional reasoning
- **Symbolic predicates** represented as tensors
- **Neural-symbolic bridging** through embedding spaces
- **Temperature-based reasoning** for controlling deterministic vs probabilistic behavior
- **Future/promise integration** for async reasoning operations

## Core Components

### 1. Tensor Class

The `tensor<T>` class provides multi-dimensional array operations:

```cpp
#include <minicoros/tensor_logic.h>

using namespace mc::tensor_logic;

// Create a 2x3 tensor
tensor<double> t({2, 3});

// Fill with values
t.fill(1.0);

// Access by flat index
double val = t[0];

// Access by multi-dimensional index
double val2 = t.at({0, 1});
```

### 2. Predicates and Knowledge Base

Predicates represent logical relations as tensors:

```cpp
knowledge_base<double> kb;

// Define a parent relation (2x2 matrix)
std::vector<double> parent_data = {1.0, 0.0, 1.0, 0.0};
tensor<double> parent({2, 2}, parent_data);

kb.add_predicate("parent", parent);

// Query predicates
auto* pred = kb.get_predicate("parent");
```

### 3. Tensor Operations

Core tensor operations for logical inference:

```cpp
using namespace mc::tensor_logic::operations;

// Matrix multiplication (logical join)
auto result = matmul(tensor_a, tensor_b);

// Element-wise multiplication (Hadamard product)
auto hadamard_result = hadamard(tensor_a, tensor_b);

// Element-wise addition
auto sum = add(tensor_a, tensor_b);

// Softmax for probabilistic reasoning
auto probabilities = softmax(logits, temperature);

// Transpose
auto transposed = transpose(matrix);
```

### 4. Reasoning Engine

The reasoning engine performs logical inference with temperature control:

```cpp
reasoning_engine<double> engine(1.0);  // temperature = 1.0

// Query knowledge base
engine.query(kb, "parent")
  .then([](tensor<double> result) -> mc::result<void> {
    // Process query result
    return {};
  })
  .done([](auto) {});

// Infer new knowledge by composition
// grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
engine.infer(kb, "parent", "parent")
  .then([](tensor<double> grandparent) -> mc::result<void> {
    // Process inference result
    return {};
  })
  .done([](auto) {});
```

### 5. Temperature-Based Reasoning

Control the reasoning mode via temperature:

- **Temperature = 0**: Deterministic, symbolic reasoning
- **Temperature > 0**: Increasingly probabilistic, neural-style reasoning

```cpp
reasoning_engine<double> deterministic(0.001);  // Near-deterministic
reasoning_engine<double> probabilistic(2.0);    // Probabilistic

// Same query with different reasoning modes
deterministic.infer(kb, "relation1", "relation2");
probabilistic.infer(kb, "relation1", "relation2");
```

### 6. Embedding Space

Bridge symbolic and neural representations:

```cpp
embedding_space<double> emb_space(128);  // 128-dimensional embeddings

// Add entity embeddings
std::vector<double> alice_embedding = {...};  // 128 dimensions
emb_space.add_entity("alice", alice_embedding);

// Compute similarity (cosine similarity)
double similarity = emb_space.similarity("alice", "bob");
```

## Example: Grandparent Rule

Classic tensor logic example - inferring grandparent from parent relations:

```cpp
reasoning_engine<double> engine;
knowledge_base<double> kb;

// Define parent relation
// Rows: Alice, Bob, Charlie
// Cols: Bob, Charlie, Dave
std::vector<double> parent_data = {
  1.0, 0.0, 0.0,  // Alice is parent of Bob
  0.0, 1.0, 0.0,  // Bob is parent of Charlie
  0.0, 0.0, 1.0   // Charlie is parent of Dave
};

tensor<double> parent({3, 3}, parent_data);
kb.add_predicate("parent", parent);

// Compute grandparent by composing parent with itself
engine.infer(kb, "parent", "parent")
  .then([](tensor<double> grandparent) -> mc::result<void> {
    // grandparent[1] shows Alice is grandparent of Charlie
    // grandparent[5] shows Bob is grandparent of Dave
    return {};
  })
  .done([](auto) {});
```

## Neural-Symbolic Integration

Combine symbolic knowledge with learned embeddings:

```cpp
// Symbolic knowledge
knowledge_base<double> kb;
std::vector<double> relation_data = {1.0, 0.5, 0.0};
tensor<double> relation({3}, relation_data);
kb.add_predicate("likes", relation);

// Neural embeddings
embedding_space<double> emb_space(3);
emb_space.add_entity("alice", {1.0, 0.0, 0.0});
emb_space.add_entity("bob", {0.0, 1.0, 0.0});
emb_space.add_entity("charlie", {0.0, 0.0, 1.0});

// Query symbolic knowledge
reasoning_engine<double> engine;
engine.query(kb, "likes")
  .then([&emb_space](tensor<double> result) -> mc::result<void> {
    // Combine with neural similarity
    double neural_similarity = emb_space.similarity("alice", "bob");
    // Use both symbolic and neural information
    return {};
  })
  .done([](auto) {});
```

## Key Features

1. **Unified Representation**: Both symbolic rules and neural operations are tensor equations
2. **Asynchronous Reasoning**: Integrated with the future/promise framework
3. **Temperature Control**: Dial between deterministic logic and probabilistic inference
4. **Composability**: Logical rules compose through tensor operations (matrix multiplication)
5. **Embedding Integration**: Bridge symbolic predicates with learned vector representations
6. **Type Safety**: Template-based design for type-safe tensor operations

## Implementation Notes

- Uses C++17 features
- No exceptions (consistent with minicoros design)
- Header-only implementation
- Compatible with both std and EASTL
- All operations work with the existing future/promise infrastructure

## Testing

Comprehensive test suite with 32 tests covering:

- Basic tensor operations
- Knowledge base management
- Tensor mathematical operations
- Reasoning engine functionality
- Embedding space operations
- Integration scenarios (grandparent rule, neural-symbolic bridging)

Run tests:
```bash
cd test
make test_tensor_logic
./test_tensor_logic
```

## References

- [Tensor Logic: The Language of AI](https://tensor-logic.org/)
- Ben Goertzel's work on neural-symbolic AI
- Pedro Domingos' Tensor Logic framework

## Future Enhancements

Potential areas for expansion:

1. Higher-order tensor operations (3D, 4D tensors)
2. Attention mechanisms as tensor operations
3. Automatic predicate invention (RESCAL-style)
4. GPU acceleration for large-scale inference
5. Probabilistic logic programming extensions
6. Integration with external tensor libraries (Eigen, Armadillo)
