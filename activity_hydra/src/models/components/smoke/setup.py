#!/usr/bin/env python

import glob
import os
import pdb
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")
    # extensions_dir = os.path.join(this_dir, "smoke", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",  # Whether a short float (float16,fp16) is supported.
            "-D__CUDA_NO_HALF_OPERATORS__",  # https://github.com/pytorch/pytorch/blob/master/cmake/Dependencies.cmake#L1117
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        raise NotImplementedError("cuda is not available")

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]
    # pdb.set_trace()

    ext_modules = [
        extension(
            "smoke._ext",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="smoke",
    version="0.1",
    author="lzccccc",
    url="https://github.com/lzccccc/SMOKE",
    description="Single-Stage Monocular 3D Object Detection via Keypoint Estimation",
    packages=find_packages(
        exclude=(
            "configs",
            "tests",
        )
    ),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
