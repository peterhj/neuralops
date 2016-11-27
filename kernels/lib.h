#ifndef __NEURALOPS_KERNELS_LIB_H__
#define __NEURALOPS_KERNELS_LIB_H__

#ifndef NEURALOPS_OMP
#define NEURALOPS_SYMBOL(name) neuralops ## _ ## name
#else
#define NEURALOPS_SYMBOL(name) neuralops_omp ## _ ## name
#endif

#ifdef min
#undef min
#endif
#define min(a, b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); (_a < _b) ? _a : _b; })

#ifdef max
#undef max
#endif
#define max(a, b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); (_a > _b) ? _a : _b; })

#endif
