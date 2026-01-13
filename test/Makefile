CXX = clang++
CXXFLAGS = -std=c++17 -fno-exceptions -I../include/ -I../tools/ -O3 -Werror -Wall -Wextra -Wpedantic

obj_files = ../tools/testing.o test_continuation_chain.o test_future.o test_operations.o
compile_duration_files = test_compile_duration.o
comparison_files = test_comparison.o
tensor_logic_files = ../tools/testing.o test_tensor_logic.o

%.o: %.cc ../include/coro.h
	$(CXX) -c $(CXXFLAGS) $< -o $@

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

test: $(obj_files)
	$(CXX) $(obj_files) -o test_main

test_compile_duration: $(compile_duration_files)
	$(CXX) $(compile_duration_files) -o test_compile_duration

comparison: $(comparison_files)
	$(CXX) $(comparison_files) -o test_comparison

test_tensor_logic: $(tensor_logic_files)
	$(CXX) $(tensor_logic_files) -o test_tensor_logic

all: test test_tensor_logic

clean:
	rm -f *.o test_main test_compile_duration test_comparison test_tensor_logic
