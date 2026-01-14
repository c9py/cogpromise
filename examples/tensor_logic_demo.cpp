/// Copyright (C) 2022 Electronic Arts Inc.  All rights reserved.
/// Example demonstrating tensor logic for neural-symbolic AI

#include <minicoros/tensor_logic.h>
#include <iostream>
#include <iomanip>

using namespace mc::tensor_logic;

void print_tensor(const std::string& name, const tensor<double>& t) {
  std::cout << name << " (";
  for (size_t i = 0; i < t.dimensions().size(); ++i) {
    std::cout << t.dimensions()[i];
    if (i < t.dimensions().size() - 1) std::cout << "x";
  }
  std::cout << "):\n";
  
  if (t.dimensions().size() == 2) {
    size_t rows = t.dimensions()[0];
    size_t cols = t.dimensions()[1];
    for (size_t i = 0; i < rows; ++i) {
      std::cout << "  [ ";
      for (size_t j = 0; j < cols; ++j) {
        std::cout << std::fixed << std::setprecision(2) << t[i * cols + j];
        if (j < cols - 1) std::cout << ", ";
      }
      std::cout << " ]\n";
    }
  } else {
    std::cout << "  [ ";
    for (size_t i = 0; i < t.size(); ++i) {
      std::cout << std::fixed << std::setprecision(2) << t[i];
      if (i < t.size() - 1) std::cout << ", ";
    }
    std::cout << " ]\n";
  }
  std::cout << std::endl;
}

void example_family_relations() {
  std::cout << "=== Example 1: Family Relations ===\n" << std::endl;
  
  // Define parent relation
  // People: Alice(0), Bob(1), Charlie(2), Dave(3)
  std::vector<double> parent_data = {
    0.0, 1.0, 0.0, 0.0,  // Alice is parent of Bob
    0.0, 0.0, 1.0, 0.0,  // Bob is parent of Charlie
    0.0, 0.0, 0.0, 1.0,  // Charlie is parent of Dave
    0.0, 0.0, 0.0, 0.0   // Dave has no children
  };
  
  tensor<double> parent({4, 4}, parent_data);
  print_tensor("Parent Relation", parent);
  
  // Create knowledge base
  knowledge_base<double> kb;
  kb.add_predicate("parent", parent);
  
  // Infer grandparent relation: grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
  reasoning_engine<double> engine;
  
  std::cout << "Inferring grandparent relation through composition...\n" << std::endl;
  
  bool completed = false;
  engine.infer(kb, "parent", "parent")
    .then([&completed](tensor<double> grandparent) -> mc::result<void> {
      print_tensor("Grandparent Relation (inferred)", grandparent);
      
      std::cout << "Interpretation:\n";
      std::cout << "  - Alice is grandparent of Charlie (position [0,2])\n";
      std::cout << "  - Bob is grandparent of Dave (position [1,3])\n";
      std::cout << std::endl;
      
      completed = true;
      return {};
    })
    .done([](auto) {});
  
  if (!completed) {
    std::cout << "ERROR: Inference failed" << std::endl;
  }
}

void example_temperature_reasoning() {
  std::cout << "=== Example 2: Temperature-Based Reasoning ===\n" << std::endl;
  
  // Create a belief network
  std::vector<double> belief_data = {
    3.0, 1.0,  // Strong belief in option 0
    1.0, 2.0   // Moderate beliefs
  };
  
  tensor<double> beliefs({2, 2}, belief_data);
  print_tensor("Raw Belief Scores", beliefs);
  
  knowledge_base<double> kb;
  kb.add_predicate("belief", beliefs);
  
  // Deterministic reasoning (low temperature)
  reasoning_engine<double> deterministic_engine(0.1);
  
  std::cout << "With LOW temperature (deterministic):\n";
  bool det_done = false;
  deterministic_engine.query(kb, "belief")
    .then([&det_done](tensor<double> result) -> mc::result<void> {
      auto softmax_result = operations::softmax(result, 0.1);
      print_tensor("Normalized beliefs", softmax_result);
      det_done = true;
      return {};
    })
    .done([](auto) {});
  
  // Probabilistic reasoning (high temperature)
  reasoning_engine<double> probabilistic_engine(2.0);
  
  std::cout << "With HIGH temperature (probabilistic):\n";
  bool prob_done = false;
  probabilistic_engine.query(kb, "belief")
    .then([&prob_done](tensor<double> result) -> mc::result<void> {
      auto softmax_result = operations::softmax(result, 2.0);
      print_tensor("Normalized beliefs", softmax_result);
      std::cout << "Note: Higher temperature produces more uniform distribution\n" << std::endl;
      prob_done = true;
      return {};
    })
    .done([](auto) {});
}

void example_embedding_space() {
  std::cout << "=== Example 3: Neural-Symbolic Integration ===\n" << std::endl;
  
  // Create embedding space
  embedding_space<double> emb_space(3);
  
  // Add entity embeddings (simplified 3D representations)
  emb_space.add_entity("cat", {1.0, 0.2, 0.1});
  emb_space.add_entity("dog", {0.9, 0.3, 0.2});
  emb_space.add_entity("fish", {0.1, 0.8, 0.9});
  emb_space.add_entity("bird", {0.2, 0.7, 0.8});
  
  std::cout << "Entity Embeddings (3D):\n";
  std::cout << "  cat:  [1.00, 0.20, 0.10]\n";
  std::cout << "  dog:  [0.90, 0.30, 0.20]\n";
  std::cout << "  fish: [0.10, 0.80, 0.90]\n";
  std::cout << "  bird: [0.20, 0.70, 0.80]\n";
  std::cout << std::endl;
  
  // Compute similarities
  std::cout << "Cosine Similarities:\n";
  std::cout << "  cat ~ dog:  " << std::fixed << std::setprecision(3) 
            << emb_space.similarity("cat", "dog") << " (high - both mammals)\n";
  std::cout << "  cat ~ fish: " << emb_space.similarity("cat", "fish") 
            << " (low - different categories)\n";
  std::cout << "  fish ~ bird: " << emb_space.similarity("fish", "bird") 
            << " (high - both non-mammals)\n";
  std::cout << "  dog ~ bird: " << emb_space.similarity("dog", "bird") 
            << " (moderate)\n";
  std::cout << std::endl;
  
  // Combine with symbolic knowledge
  std::vector<double> is_pet_data = {1.0, 1.0, 0.5, 0.8};  // likelihood of being a pet
  tensor<double> is_pet({4}, is_pet_data);
  
  knowledge_base<double> kb;
  kb.add_predicate("is_pet", is_pet);
  
  reasoning_engine<double> engine;
  
  std::cout << "Symbolic Knowledge: is_pet relation\n";
  bool done = false;
  engine.query(kb, "is_pet")
    .then([&done](tensor<double> pet_scores) -> mc::result<void> {
      std::cout << "  cat:  " << std::fixed << std::setprecision(2) << pet_scores[0] << "\n";
      std::cout << "  dog:  " << pet_scores[1] << "\n";
      std::cout << "  fish: " << pet_scores[2] << "\n";
      std::cout << "  bird: " << pet_scores[3] << "\n";
      std::cout << "\nCombining neural similarities with symbolic is_pet knowledge\n";
      std::cout << "enables hybrid reasoning about animal relationships!\n";
      done = true;
      return {};
    })
    .done([](auto) {});
  
  std::cout << std::endl;
}

void example_matrix_operations() {
  std::cout << "=== Example 4: Tensor Operations ===\n" << std::endl;
  
  std::vector<double> a_data = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> b_data = {2.0, 0.0, 1.0, 3.0};
  
  tensor<double> a({2, 2}, a_data);
  tensor<double> b({2, 2}, b_data);
  
  print_tensor("Matrix A", a);
  print_tensor("Matrix B", b);
  
  // Matrix multiplication (logical join)
  auto matmul_result = operations::matmul(a, b);
  print_tensor("A × B (Matrix Multiplication)", matmul_result);
  
  // Hadamard product (element-wise multiplication)
  auto hadamard_result = operations::hadamard(a, b);
  print_tensor("A ⊙ B (Hadamard Product)", hadamard_result);
  
  // Addition
  auto add_result = operations::add(a, b);
  print_tensor("A + B (Element-wise Addition)", add_result);
  
  // Transpose
  auto transpose_result = operations::transpose(a);
  print_tensor("A^T (Transpose)", transpose_result);
}

int main() {
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║     Tensor Logic for Neural-Symbolic AI - Examples      ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";
  std::cout << "\n";
  
  example_family_relations();
  std::cout << "────────────────────────────────────────────────────────\n\n";
  
  example_temperature_reasoning();
  std::cout << "────────────────────────────────────────────────────────\n\n";
  
  example_embedding_space();
  std::cout << "────────────────────────────────────────────────────────\n\n";
  
  example_matrix_operations();
  
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║                    Examples Complete                     ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";
  std::cout << "\n";
  
  return 0;
}
