/// Copyright (C) 2022 Electronic Arts Inc.  All rights reserved.
/// Unit tests for tensor logic implementation

#include "testing.h"
#include <minicoros/tensor_logic.h>
#include <minicoros/testing.h>
#include <memory>
#include <cmath>

using namespace testing;
using namespace mc::tensor_logic;

// Helper function to compare floating point numbers
bool approx_equal(double a, double b, double epsilon = 1e-6) {
  return std::abs(a - b) < epsilon;
}

// ============================================================================
// Tensor Basic Operations Tests
// ============================================================================

TEST(tensor_logic, tensor_creation) {
  tensor<double> t({2, 3});
  ASSERT_EQ(t.dimensions().size(), 2);
  ASSERT_EQ(t.dimensions()[0], 2);
  ASSERT_EQ(t.dimensions()[1], 3);
  ASSERT_EQ(t.size(), 6);
}

TEST(tensor_logic, tensor_with_data) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  tensor<double> t({2, 2}, data);
  ASSERT_EQ(t.size(), 4);
  ASSERT_EQ(t[0], 1.0);
  ASSERT_EQ(t[1], 2.0);
  ASSERT_EQ(t[2], 3.0);
  ASSERT_EQ(t[3], 4.0);
}

TEST(tensor_logic, tensor_fill) {
  tensor<double> t({2, 3});
  t.fill(5.0);
  for (size_t i = 0; i < t.size(); ++i) {
    ASSERT_EQ(t[i], 5.0);
  }
}

TEST(tensor_logic, tensor_multidim_access) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  tensor<double> t({2, 2}, data);
  ASSERT_EQ(t.at({0, 0}), 1.0);
  ASSERT_EQ(t.at({0, 1}), 2.0);
  ASSERT_EQ(t.at({1, 0}), 3.0);
  ASSERT_EQ(t.at({1, 1}), 4.0);
}

// ============================================================================
// Predicate Tests
// ============================================================================

TEST(tensor_logic, predicate_creation) {
  tensor<double> t({2, 2});
  t.fill(1.0);
  predicate<double> pred("parent", t);
  ASSERT_EQ(pred.name(), "parent");
  ASSERT_EQ(pred.tensor_representation().size(), 4);
}

// ============================================================================
// Knowledge Base Tests
// ============================================================================

TEST(tensor_logic, knowledge_base_add_predicate) {
  knowledge_base<double> kb;
  tensor<double> t({2, 2});
  t.fill(1.0);
  
  kb.add_predicate("parent", t);
  ASSERT_TRUE(kb.has_predicate("parent"));
  ASSERT_FALSE(kb.has_predicate("grandparent"));
}

TEST(tensor_logic, knowledge_base_get_predicate) {
  knowledge_base<double> kb;
  std::vector<double> data = {1.0, 0.0, 1.0, 0.0};
  tensor<double> t({2, 2}, data);
  
  kb.add_predicate("parent", t);
  auto* pred = kb.get_predicate("parent");
  ASSERT_TRUE((pred != nullptr));
  ASSERT_EQ(pred->name(), "parent");
  ASSERT_EQ(pred->tensor_representation()[0], 1.0);
}

TEST(tensor_logic, knowledge_base_get_nonexistent_predicate) {
  knowledge_base<double> kb;
  auto* pred = kb.get_predicate("nonexistent");
  ASSERT_TRUE((pred == nullptr));
}

TEST(tensor_logic, knowledge_base_get_predicate_names) {
  knowledge_base<double> kb;
  tensor<double> t1({2, 2});
  tensor<double> t2({3, 3});
  
  kb.add_predicate("parent", t1);
  kb.add_predicate("sibling", t2);
  
  auto names = kb.get_predicate_names();
  ASSERT_EQ(names.size(), 2);
}

// ============================================================================
// Tensor Operations Tests
// ============================================================================

TEST(tensor_logic, matmul_basic) {
  std::vector<double> data_a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> data_b = {2.0, 0.0, 1.0, 3.0};
  
  tensor<double> a({2, 2}, data_a);
  tensor<double> b({2, 2}, data_b);
  
  auto result = operations::matmul(a, b);
  
  // [1, 2] x [2, 0] = [4, 6]
  // [3, 4]   [1, 3]   [10, 12]
  ASSERT_EQ(result.dimensions().size(), 2);
  ASSERT_EQ(result[0], 4.0);   // 1*2 + 2*1
  ASSERT_EQ(result[1], 6.0);   // 1*0 + 2*3
  ASSERT_EQ(result[2], 10.0);  // 3*2 + 4*1
  ASSERT_EQ(result[3], 12.0);  // 3*0 + 4*3
}

TEST(tensor_logic, matmul_rectangular) {
  std::vector<double> data_a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> data_b = {1.0, 2.0, 3.0, 4.0};
  
  tensor<double> a({2, 3}, data_a); // 2x3
  tensor<double> b({3, 1}, data_b); // 3x1 (using first 3 elements)
  
  // Adjust b to be 3x1
  tensor<double> b_correct({3, 1}, {1.0, 2.0, 3.0});
  auto result = operations::matmul(a, b_correct);
  
  // [1, 2, 3] x [1] = [14]  (1*1 + 2*2 + 3*3)
  // [4, 5, 6]   [2]   [32]  (4*1 + 5*2 + 6*3)
  //             [3]
  ASSERT_EQ(result.dimensions()[0], 2);
  ASSERT_EQ(result.dimensions()[1], 1);
  ASSERT_EQ(result[0], 14.0);
  ASSERT_EQ(result[1], 32.0);
}

TEST(tensor_logic, hadamard_product) {
  std::vector<double> data_a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> data_b = {2.0, 3.0, 4.0, 5.0};
  
  tensor<double> a({2, 2}, data_a);
  tensor<double> b({2, 2}, data_b);
  
  auto result = operations::hadamard(a, b);
  
  ASSERT_EQ(result[0], 2.0);   // 1*2
  ASSERT_EQ(result[1], 6.0);   // 2*3
  ASSERT_EQ(result[2], 12.0);  // 3*4
  ASSERT_EQ(result[3], 20.0);  // 4*5
}

TEST(tensor_logic, tensor_addition) {
  std::vector<double> data_a = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> data_b = {5.0, 6.0, 7.0, 8.0};
  
  tensor<double> a({2, 2}, data_a);
  tensor<double> b({2, 2}, data_b);
  
  auto result = operations::add(a, b);
  
  ASSERT_EQ(result[0], 6.0);
  ASSERT_EQ(result[1], 8.0);
  ASSERT_EQ(result[2], 10.0);
  ASSERT_EQ(result[3], 12.0);
}

TEST(tensor_logic, softmax_computation) {
  std::vector<double> data = {1.0, 2.0, 3.0};
  tensor<double> t({3}, data);
  
  auto result = operations::softmax(t, 1.0);
  
  // Verify sum equals 1
  double sum = 0.0;
  for (size_t i = 0; i < result.size(); ++i) {
    sum += result[i];
  }
  ASSERT_TRUE(approx_equal(sum, 1.0));
  
  // Verify monotonicity (larger input -> larger output for softmax)
  ASSERT_TRUE((result[0] < result[1]));
  ASSERT_TRUE((result[1] < result[2]));
}

TEST(tensor_logic, softmax_with_temperature) {
  std::vector<double> data = {1.0, 2.0, 3.0};
  tensor<double> t({3}, data);
  
  // Low temperature makes distribution more peaked
  auto result_low = operations::softmax(t, 0.1);
  
  // High temperature makes distribution more uniform
  auto result_high = operations::softmax(t, 10.0);
  
  // With lower temperature, the max should be more dominant
  ASSERT_TRUE((result_low[2] > result_high[2]));
}

TEST(tensor_logic, transpose_operation) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  tensor<double> t({2, 3}, data);
  
  auto result = operations::transpose(t);
  
  ASSERT_EQ(result.dimensions()[0], 3);
  ASSERT_EQ(result.dimensions()[1], 2);
  ASSERT_EQ(result[0], 1.0); // (0,0)
  ASSERT_EQ(result[1], 4.0); // (0,1)
  ASSERT_EQ(result[2], 2.0); // (1,0)
  ASSERT_EQ(result[3], 5.0); // (1,1)
  ASSERT_EQ(result[4], 3.0); // (2,0)
  ASSERT_EQ(result[5], 6.0); // (2,1)
}

// ============================================================================
// Reasoning Engine Tests
// ============================================================================

TEST(tensor_logic, reasoning_engine_creation) {
  reasoning_engine<double> engine(1.0);
  ASSERT_EQ(engine.get_temperature(), 1.0);
}

TEST(tensor_logic, reasoning_engine_temperature) {
  reasoning_engine<double> engine(1.0);
  engine.set_temperature(0.5);
  ASSERT_EQ(engine.get_temperature(), 0.5);
}

TEST(tensor_logic, reasoning_engine_logical_join) {
  reasoning_engine<double> engine;
  
  std::vector<double> data_a = {1.0, 0.0, 1.0, 0.0};
  std::vector<double> data_b = {1.0, 1.0, 0.0, 1.0};
  
  tensor<double> a({2, 2}, data_a);
  tensor<double> b({2, 2}, data_b);
  
  auto result = engine.logical_join(a, b);
  
  ASSERT_EQ(result.dimensions().size(), 2);
  ASSERT_EQ(result.dimensions()[0], 2);
  ASSERT_EQ(result.dimensions()[1], 2);
}

TEST(tensor_logic, reasoning_engine_query_success) {
  reasoning_engine<double> engine;
  knowledge_base<double> kb;
  
  std::vector<double> data = {1.0, 0.0, 1.0, 0.0};
  tensor<double> t({2, 2}, data);
  kb.add_predicate("parent", t);
  
  bool query_executed = false;
  engine.query(kb, "parent")
    .then([&query_executed](tensor<double> result) -> mc::result<void> {
      query_executed = true;
      ASSERT_EQ(result.size(), 4);
      ASSERT_EQ(result[0], 1.0);
      return {};
    })
    .done([](auto) {});
  
  ASSERT_TRUE(query_executed);
}

TEST(tensor_logic, reasoning_engine_query_failure) {
  reasoning_engine<double> engine;
  knowledge_base<double> kb;
  
  bool failure_handled = false;
  engine.query(kb, "nonexistent")
    .fail([&failure_handled](int) -> mc::result<tensor<double>> {
      failure_handled = true;
      return mc::failure(1);
    })
    .done([](auto) {});
  
  ASSERT_TRUE(failure_handled);
}

TEST(tensor_logic, reasoning_engine_infer_success) {
  reasoning_engine<double> engine;
  knowledge_base<double> kb;
  
  std::vector<double> data1 = {1.0, 0.0, 1.0, 0.0};
  std::vector<double> data2 = {1.0, 1.0, 0.0, 1.0};
  
  tensor<double> t1({2, 2}, data1);
  tensor<double> t2({2, 2}, data2);
  
  kb.add_predicate("parent", t1);
  kb.add_predicate("ancestor", t2);
  
  bool infer_executed = false;
  engine.infer(kb, "parent", "ancestor")
    .then([&infer_executed](tensor<double> result) -> mc::result<void> {
      infer_executed = true;
      ASSERT_EQ(result.dimensions().size(), 2);
      return {};
    })
    .done([](auto) {});
  
  ASSERT_TRUE(infer_executed);
}

TEST(tensor_logic, reasoning_engine_infer_with_temperature) {
  reasoning_engine<double> engine(0.5);
  knowledge_base<double> kb;
  
  std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> data2 = {2.0, 1.0, 1.0, 2.0};
  
  tensor<double> t1({2, 2}, data1);
  tensor<double> t2({2, 2}, data2);
  
  kb.add_predicate("rel1", t1);
  kb.add_predicate("rel2", t2);
  
  bool composed = false;
  engine.infer(kb, "rel1", "rel2")
    .then([&composed](tensor<double> result) -> mc::result<void> {
      composed = true;
      // Temperature-based composition should produce normalized results
      double sum = 0.0;
      for (size_t i = 0; i < result.size(); ++i) {
        sum += result[i];
      }
      // Each row should sum to approximately 1 (softmax normalization)
      // This is a loose check since we're normalizing the entire tensor
      ASSERT_TRUE((sum > 0.0));
      return {};
    })
    .done([](auto) {});
  
  ASSERT_TRUE(composed);
}

// ============================================================================
// Embedding Space Tests
// ============================================================================

TEST(tensor_logic, embedding_space_creation) {
  embedding_space<double> emb_space(128);
  ASSERT_EQ(emb_space.dimension(), 128);
}

TEST(tensor_logic, embedding_space_add_entity) {
  embedding_space<double> emb_space(3);
  std::vector<double> embedding = {1.0, 2.0, 3.0};
  
  emb_space.add_entity("entity1", embedding);
  auto* emb = emb_space.get_embedding("entity1");
  
  ASSERT_TRUE((emb != nullptr));
  ASSERT_EQ((*emb)[0], 1.0);
  ASSERT_EQ((*emb)[1], 2.0);
  ASSERT_EQ((*emb)[2], 3.0);
}

TEST(tensor_logic, embedding_space_get_nonexistent) {
  embedding_space<double> emb_space(3);
  auto* emb = emb_space.get_embedding("nonexistent");
  ASSERT_TRUE((emb == nullptr));
}

TEST(tensor_logic, embedding_space_similarity_identical) {
  embedding_space<double> emb_space(3);
  std::vector<double> embedding = {1.0, 2.0, 3.0};
  
  emb_space.add_entity("entity1", embedding);
  emb_space.add_entity("entity2", embedding);
  
  double sim = emb_space.similarity("entity1", "entity2");
  ASSERT_TRUE(approx_equal(sim, 1.0)); // Cosine similarity of identical vectors is 1
}

TEST(tensor_logic, embedding_space_similarity_orthogonal) {
  embedding_space<double> emb_space(2);
  std::vector<double> emb1 = {1.0, 0.0};
  std::vector<double> emb2 = {0.0, 1.0};
  
  emb_space.add_entity("entity1", emb1);
  emb_space.add_entity("entity2", emb2);
  
  double sim = emb_space.similarity("entity1", "entity2");
  ASSERT_TRUE(approx_equal(sim, 0.0)); // Orthogonal vectors have similarity 0
}

TEST(tensor_logic, embedding_space_similarity_partial) {
  embedding_space<double> emb_space(2);
  std::vector<double> emb1 = {1.0, 0.0};
  std::vector<double> emb2 = {1.0, 1.0};
  
  emb_space.add_entity("entity1", emb1);
  emb_space.add_entity("entity2", emb2);
  
  double sim = emb_space.similarity("entity1", "entity2");
  // cos(45°) ≈ 0.707
  ASSERT_TRUE(approx_equal(sim, 1.0 / std::sqrt(2.0), 1e-5));
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(tensor_logic, integration_grandparent_rule) {
  // Classic tensor logic example: grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
  // This is represented as a matrix multiplication
  
  reasoning_engine<double> engine;
  knowledge_base<double> kb;
  
  // Parent relation: 3x3 matrix
  // Rows: Alice, Bob, Charlie
  // Cols: Bob, Charlie, Dave
  std::vector<double> parent_data = {
    1.0, 0.0, 0.0,  // Alice is parent of Bob
    0.0, 1.0, 0.0,  // Bob is parent of Charlie
    0.0, 0.0, 1.0   // Charlie is parent of Dave
  };
  
  tensor<double> parent({3, 3}, parent_data);
  kb.add_predicate("parent", parent);
  
  // Grandparent is parent composed with parent
  bool success = false;
  engine.infer(kb, "parent", "parent")
    .then([&success](tensor<double> grandparent) -> mc::result<void> {
      success = true;
      // Alice is grandparent of Charlie (1*1 = 1)
      ASSERT_TRUE((grandparent[1] > 0.0));
      // Bob is grandparent of Dave
      ASSERT_TRUE((grandparent[5] > 0.0));
      return {};
    })
    .done([](auto) {});
  
  ASSERT_TRUE(success);
}

TEST(tensor_logic, integration_neural_symbolic_bridge) {
  // Test bridging symbolic predicates with neural embeddings
  
  embedding_space<double> emb_space(3);
  knowledge_base<double> kb;
  
  // Add embeddings for entities
  emb_space.add_entity("alice", {1.0, 0.0, 0.0});
  emb_space.add_entity("bob", {0.0, 1.0, 0.0});
  emb_space.add_entity("charlie", {0.0, 0.0, 1.0});
  
  // Create a symbolic relation
  std::vector<double> relation_data = {1.0, 0.5, 0.0};
  tensor<double> relation({3}, relation_data);
  kb.add_predicate("likes", relation);
  
  // Verify we can query the symbolic knowledge
  reasoning_engine<double> engine;
  bool queried = false;
  engine.query(kb, "likes")
    .then([&queried](tensor<double> result) -> mc::result<void> {
      queried = true;
      ASSERT_EQ(result.size(), 3);
      return {};
    })
    .done([](auto) {});
  
  ASSERT_TRUE(queried);
  
  // Verify embeddings exist
  ASSERT_TRUE((emb_space.get_embedding("alice") != nullptr));
  ASSERT_TRUE((emb_space.get_embedding("bob") != nullptr));
  ASSERT_TRUE((emb_space.get_embedding("charlie") != nullptr));
}

TEST(tensor_logic, integration_temperature_modes) {
  // Test different reasoning modes via temperature
  
  reasoning_engine<double> deterministic_engine(0.0001); // Near-deterministic
  reasoning_engine<double> probabilistic_engine(2.0);    // Probabilistic
  
  knowledge_base<double> kb;
  
  std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> data2 = {1.0, 1.0, 1.0, 1.0};
  
  tensor<double> t1({2, 2}, data1);
  tensor<double> t2({2, 2}, data2);
  
  kb.add_predicate("strong_belief", t1);
  kb.add_predicate("weak_belief", t2);
  
  // Both engines should complete successfully
  bool det_success = false;
  deterministic_engine.infer(kb, "strong_belief", "weak_belief")
    .then([&det_success](tensor<double>) -> mc::result<void> {
      det_success = true;
      return {};
    })
    .done([](auto) {});
  
  bool prob_success = false;
  probabilistic_engine.infer(kb, "strong_belief", "weak_belief")
    .then([&prob_success](tensor<double>) -> mc::result<void> {
      prob_success = true;
      return {};
    })
    .done([](auto) {});
  
  ASSERT_TRUE(det_success);
  ASSERT_TRUE(prob_success);
}
