import os
import sys
import subprocess

from setuptools import find_packages, setup
from setuptools.command.install import install
# TODO: This is a bit buggy since it requires torch before installing torch.
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext




class PostInstallation(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # Note: buggy for kornia==0.5.3 and it will be fixed in the next version.
        # Set kornia to 0.5.2 temporarily
        subprocess.call([sys.executable, '-m', 'pip', 'install', 'kornia==0.5.2', '--no-dependencies'])


if __name__ == '__main__':
    version = '0.1.0'
    setup(
        name='clocs_lqs',
        version=version,
        description='clocs implementation',
        install_requires=[
            'numpy',
            'torch',
            'numba',
            'easydict',
            'kornia',
            'pyyaml'
        ],
        author='Qingsi Lai',
        author_email='laiqingsi@outlook.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
            'install': PostInstallation,
            # Post installation cannot be done. ref: https://github.com/pypa/setuptools/issues/1936.
            # 'develop': PostInstallation,
        },
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='pcdet.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='pcdet.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ]
            ),
        ],
    )
