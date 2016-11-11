#ifndef __NEURALOPS_KERNELS_LIB_H__
#define __NEURALOPS_KERNELS_LIB_H__

#ifndef NEURALOPS_OMP
#define NEURALOPS_SYMBOL(name) neuralops ## _ ## name
#else
#define NEURALOPS_SYMBOL(name) neuralops_omp ## _ ## name
#endif

#endif
