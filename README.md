# MVOp

MVOp is a high-performance C++ library for **vector and matrix operations** that **requires CUDA**.  
All computations are performed on the GPU, leveraging CUDA for maximum performance on large-scale linear algebra tasks. The API is designed to be clean and easy to use for GPU-accelerated workflows.

---

## Features

- **Vector and Matrix Operations:** Efficient GPU implementations for addition, multiplication, dot product, norms, and more.
- **CUDA-Only:** All operations require a CUDA-capable GPU and NVIDIA CUDA Toolkit.
- **CMake Presets:** Easy configuration for CUDA builds.
- **Installable:** Standard CMake install targets for easy integration.

---

## Build & Targets

### Configure 
    $ cmake -S <source_dir> -B <build_dir>

You can use `presets`

    $ cmake -S <source_dir> --preset <preset_name>

To know the existing presets

    $ cmake -S <source_dir> --list-presets

### Build
    $ cmake --build <build_dir>

## Run Tests

Tests are registered with CTest via `gtest_discover_tests()`. The command to run them depends on the CMake generator you configured with.

### For all generators (recommended)

Using `ctest` directly works regardless of the generator:

    $ ctest --test-dir <build_dir> -C <config>

Where `<config>` is `Release` or `Debug` (with Visual Studio multi-config generators). Example using the preset build directory:

    $ ctest --test-dir <build_dir> -C Release

### For single-config generators (Makefiles, Ninja)

If you configured with a single-config generator, the `test` build target is available:

    $ cmake --build <build_dir> --target test

### List available tests

    $ ctest --test-dir <build_dir> -C Release -N

### Run all tests in one executable (e.g. all Test_Vector_Add tests)

    $ ctest --test-dir <build_dir> -C Release -R Test_Vector_Add

### Run a specific test case (regex match on test name)

For a single registered test case (e.g. `Test_Vector_Add.AddFloats`), use the `-R` regex filter:

    $ ctest --test-dir <build_dir> -C Release -R "Test_Vector_Add.AddFloats"

> Note: `gtest_discover_tests()` makes each individual `TEST()`/`TEST_F()` case a separate CTest entry (named `ExeName.TestCaseName`), so `-R` matches against those when using `ctest`.

### Run tests with verbose output

    $ ctest --test-dir <build_dir> -C Release -V

## Install
    $ cmake --install <build_dir> --prefix <install_dir>

---

## Requirements

- NVIDIA CUDA-capable GPU
- CUDA Toolkit (tested with CUDA 11+)
- CMake 3.18 or newer
- C++20 compiler (MSVC, GCC, or Clang with CUDA support)

---

## License

MIT License (or your chosen license)