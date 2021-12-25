from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='MarchingCube',
    ext_modules=[
        CUDAExtension('MarchingCube_cuda', [
            'src/MarchingCube_api.cpp',

            'src/MarchingCube_cpp.cpp'
            'src/common_cu.cu',
            'src/MarchingCube_cu.cu',
            'src/thrustWrapper_cu.cu',
        ], extra_compile_args={'cxx': ['-g'],
                               'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
