#ifndef ARCH_FLAGS_IS_DEF
#define ARCH_FLAGS_IS_DEF

#include <stdint.h>

#define IS_LITTLE_ENDIAN                                                       \
  ((union {                                                                    \
     uint32_t u;                                                               \
     uint8_t c;                                                                \
   }){.u = 1}                                                                  \
       .c)
#define IS_BIG_ENDIAN (!IS_LITTLE_ENDIAN)

#ifdef ENABLE_VECTO

#define AVX_VEC_SIZE_CHAR 32
#define AVX_VEC_SIZE_INT 8
#define AVX_VEC_SIZE_FLOAT 8
#define AVX_VEC_SIZE_DOUBLE 4

#define AVX_WIDTH AVX_VEC_SIZE_CHAR

#define AVX512_VEC_SIZE_CHAR 64
#define AVX512_VEC_SIZE_INT 16
#define AVX512_VEC_SIZE_FLOAT 16
#define AVX512_VEC_SIZE_DOUBLE 8

#define AVX512_WIDTH AVX512_VEC_SIZE_CHAR

#define SSE_VEC_SIZE_CHAR 16
#define SSE_VEC_SIZE_INT 4
#define SSE_VEC_SIZE_FLOAT 4
#define SSE_VEC_SIZE_DOUBLE 2

#define SSE_WIDTH SSE_VEC_SIZE_CHAR

#endif

void arch_flags_print (void);

#endif
