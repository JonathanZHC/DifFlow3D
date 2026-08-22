from pathlib import Path
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

sources = [
    'src/pointnet2_api.cpp',
    'src/ball_query.cpp',
    'src/ball_query_gpu.cu',
    'src/group_points.cpp',
    'src/group_points_gpu.cu',
    'src/interpolate.cpp',
    'src/interpolate_gpu.cu',
    'src/sampling.cpp',
    'src/sampling_gpu.cu',
    'src/gaussian_recovery.cpp',
    'src/gaussian_recovery_gpu.cu',
    'src/anchor_motion.cpp',
    'src/anchor_motion_gpu.cu',
]

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            'pointnet2_cuda',
            sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},
)
