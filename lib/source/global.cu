#pragma once

#include "matrix_vector_op.hpp"
#include <cuda_runtime.h>

__constant__ unsigned char CudaConstMem_ptr[USED_CONST_MEM_BYTES];
