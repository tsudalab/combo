from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
from Cython.Build import cythonize

compile_flags =['-O3',]
ext_mods = [Extension( name ='combo.misc._src.traceAB',
                      sources=['combo/misc/_src/traceAB.pyx'],
                      include_dirs=[numpy.get_include()],
                      extra_compile_args = compile_flags),
            Extension( name ='combo.misc._src.cholupdate',
                       sources=['combo/misc/_src/cholupdate.pyx'],
                       include_dirs=[numpy.get_include()],
                       extra_compile_args = compile_flags),
            Extension( name ='combo.misc._src.diagAB',
                       sources=['combo/misc/_src/diagAB.pyx'],
                       include_dirs=[numpy.get_include()],
                       extra_compile_args = compile_flags),
            Extension( name ='combo.gp.cov._src.enhance_gauss',
                       sources=['combo/gp/cov/_src/enhance_gauss.pyx'],
                       include_dirs=[numpy.get_include()],
                       extra_compile_args = compile_flags),
            Extension( name ='combo.misc._src.logsumexp',
                       sources=['combo/misc/_src/logsumexp.pyx'],
                       include_dirs=[numpy.get_include()],
                       extra_compile_args = compile_flags )
            ]
setup(
    name = 'combo',
    version = '0.2',
    author = 'Tsuyoshi Ueno',
    author_email = "tsuyoshi.ueno@gmail.com",
    packages = ['combo','combo.misc','combo.misc._src',
    'combo.gp', 'combo.gp.cov','combo.gp.cov._src', 'combo.gp.mean','combo.gp.core','combo.gp.inf',
    'combo.gp.lik', 'combo.opt', 'combo.blm.lik','combo.blm.prior','combo.blm.basis', 'combo.blm.inf','combo.blm.core',
    'combo.blm.lik._src','combo.blm', 'combo.search', 'combo.search.discrete'],
    package_dir={'combo': 'combo'},
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_mods
)
