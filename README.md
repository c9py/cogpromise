`Minicoros` is a C++17 header-only library that implements future chains (similar to coroutines). Heavily inspired by Denis Blank's (Naios) Continuable library but with the following differences:
* __Faster compilation time__ through simpler code:
  * Minicoros executes two calls to `operator new` for each `.then` handler, as opposed to the Continuable library that opts for zero-cost abstractions. Allocations can possibly be mitigated by using pooled allocators and `std::function` implementations that have larger SBO,
  but no such support exists in Minicoros
  * Less flexibility in values accepted to/from callbacks
* __More opinionated__, which should make it easier to use
* No threading support, no exceptions, uses `std::function`

Why use Minicoros over Continuables? Minicoros is much friendlier to the compiler; preliminary measurements point to code using Minicoros compiling in 1/2 to 1/4 of the time Continuable uses and that Minicoros scales _much_ better for longer chains. Compiler memory usage follows a similar pattern. (TODO: measure)

Unfortunately there's no support for C++20 coroutines yet, but that will be added.

## Tensor Logic Extension

This library now includes a comprehensive **Tensor Logic** implementation that bridges neural and symbolic AI. Tensor Logic represents logical rules as tensor equations, enabling:

- **Symbolic reasoning** through tensor operations
- **Neural-symbolic integration** via embedding spaces
- **Temperature-based reasoning** (deterministic ↔ probabilistic)
- **Asynchronous inference** using the future/promise framework

See [TENSOR_LOGIC.md](TENSOR_LOGIC.md) for detailed documentation and examples.

Quick example:
```cpp
#include <minicoros/tensor_logic.h>

using namespace mc::tensor_logic;

// Create knowledge base
knowledge_base<double> kb;
tensor<double> parent({3, 3}, parent_data);
kb.add_predicate("parent", parent);

// Infer grandparent rule: grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
reasoning_engine<double> engine;
engine.infer(kb, "parent", "parent")
  .then([](tensor<double> grandparent) -> mc::result<void> {
    // Use inferred grandparent relation
    return {};
  })
  .done([](auto) {});
```

## Examples
```cpp
mc::future<int> sum1(int o1, int o2) {
  // Return a future that synchronously/immediately returns a successful value
  return mc::make_successful_future<int>(o1 + o2);
}

mc::future<int> sum2(int o1, int o2) {
  // A future is backed by a promise -- the promise resolves the future
  return mc::future<int>([o1, o2] (mc::promise<int> promise) {
    // This lambda will get invoked lazily at a later point
    promise(o1 + o2);
  });
}

void main() {
  sum1(4, 10)
    .then([](int sum) -> mc::result<int> {  // Specify the return type
      if (sum != 14) {
        // Preferred way of returning an error. With `failure`, you don't have to re-type the return type
        return mc::failure(EBADSUM);
      }

      // ... or by returning a future that fails:
      if (sum != 14)
        return mc::make_failed_future<int>(EBADSUM);

      return sum * 2;
    })
    .fail([](int error) {
      return failure(error);  // Rethrow
    })
    .fail([](int error) -> mc::result<void> {  // Explicitly recover from the error
      return {};
    })
    .then([] () -> mc::result<int, int> {
      return sum1(1, 3) && sum2(1, 3);
    })
    .then([] (int s1, int s2) {
      // Actually, it's alright not specifying mc::result on .thens that return void
      ASSERT_EQ(s1, s2);
    })
   .ignore_result(); // To protect against forgetting to handle errors, a chain either needs to be returned
                         // from the current function, or finalized with `done` 
}
```

## Contributing
Before you can contribute, EA must have a Contributor License Agreement (CLA) on file that has been signed by each contributor.
You can sign here: [Go to CLA](https://electronicarts.na1.echosign.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhByHRvZqmltGtliuExmuV-WNzlaJGPhbSRg2ufuPsM3P0QmILZjLpkGslg24-UJtek*)

