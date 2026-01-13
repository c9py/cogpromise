CXX = clang++
CXXFLAGS = -std=c++17 -fno-exceptions -I../include/ -O2 -Wall -Wextra -Wpedantic

tensor_logic_demo: tensor_logic_demo.cpp
	$(CXX) $(CXXFLAGS) tensor_logic_demo.cpp -o tensor_logic_demo

clean:
	rm -f tensor_logic_demo

.PHONY: clean
