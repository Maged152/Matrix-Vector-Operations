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

## How to Use

1. **Clone the repository.**
2. **Build with CMake:**  
   ```
   cmake -S <source_dir> -B <build_dir>
   cmake --build <build_dir>
   ```
3. **Install:**  
   ```
   cmake --install <build_dir> --prefix <install_dir>
   ```

---

## Requirements

- NVIDIA CUDA-capable GPU
- CUDA Toolkit (tested with CUDA 11+)
- CMake 3.18 or newer
- C++20 compiler (MSVC, GCC, or Clang with CUDA support)

---

## License

MIT License (or your chosen license)