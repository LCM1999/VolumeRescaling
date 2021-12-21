#include <torch/extension.h>

#include "MarchingCube_h.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("marching_cube", &marching_cube, "marching_cube");
    m.def("cudaMarchingCube", &cudaMarchingCube, "cudaMarchingCube");
}