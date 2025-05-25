# MVOp
MVOp is a comprehensive template designed to streamline the process of building a library using `CMake`. This template provides a solid foundation, including a clear and organized directory structure, essential CMake configuration files, and best practices for library development. With MVOp, developers can quickly set up a new project with consistent build settings and focus on writing their library code, rather than spending time on initial configuration. Whether you are developing a static or shared library, this template ensures a smooth and efficient start to your CMake-based project.

# Features
* Build the Library: Easily compile your library with CMake.
* Examples: Include example code to demonstrate how to use your library.
* Tests Using Google Test: Integrate unit tests using Google Test framework to ensure your library works as expected.
* Documentation Using Doxygen: Generate professional documentation for your codebase using Doxygen.
* Install: Install your library and its headers to a specified location.

# How to Use
* Clone the repository.
* Rename the project:
   * Replace all instances of `MVOp` with your project name.
* Make sure your project follows the same directory structure.

# Build & Targets

## Configure 
    $ cmake -S <source_dir> -B <build_dir>

You can use `presets`

    $ cmake -S <source_dir> --preset <preset_name>

To know the existing presets

    $ cmake -S <source_dir> --list-presets


## Build
    $ cmake --build <build_dir>

## Install
    $ cmake --install <build_dir> --prefix <install_dir>

## Generate Documentations
    $ cmake --build <build_dir> --target documentation

## Running Tests
### Run a specific test
    $ cmake --build <build_dir> --target Test_<test_name>

### Run all tests
    $ ctest --test-dir <build_dir>

## Examples
    $ cmake --build <build_dir> --target Example_<example_name>
