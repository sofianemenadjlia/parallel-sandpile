#include "kernel/ocl/common.cl"

__kernel void ssandPile_ocl (__global unsigned *in, __global unsigned *out)
{
  int y = get_global_id (1);
  int x = get_global_id (0);

  __local unsigned tile[GPU_TILE_H][GPU_TILE_W];

  unsigned local_y = get_local_id (1);
  unsigned local_x = get_local_id (0);

  if(x > 0 && y > 0 && x < DIM-1 && y < DIM-1)
  {

    tile[local_y][local_x] = 
    (in[y * DIM + x] % 4) + 
    (in[(y+1) * DIM + x] /4) + 
    (in[(y-1) * DIM + x] /4) + 
    (in[y * DIM + (x+1)] /4) + 
    (in[y * DIM + (x-1)] /4);
    
  barrier (CLK_LOCAL_MEM_FENCE);

  out [y * DIM + x] = tile[local_y][local_x];
  }
}

__kernel void ssandPile_ocl_hybrid (__global unsigned *in, __global unsigned *out, __global unsigned *new, unsigned offset, unsigned iteration)
{
  int y = get_global_id (1) + offset;
  int x = get_global_id (0);

  int index = y * DIM + x;

  unsigned local_y = get_local_id (1);
  unsigned local_x = get_local_id (0);

  if(x > 0 && y > 0 && x < DIM-1 && y < DIM-1)
  {

    out[index] = 
    (in[y * DIM + x] % 4) + 
    (in[(y+1) * DIM + x] /4) + 
    (in[(y-1) * DIM + x] /4) + 
    (in[y * DIM + (x+1)] /4) + 
    (in[y * DIM + (x-1)] /4);
    
  barrier (CLK_LOCAL_MEM_FENCE);

  y = y / GPU_TILE_H;
  x = x / GPU_TILE_W;
  int tile_index = y * (DIM / GPU_TILE_W) + x;
  if (iteration % (DIM/8) == 0)
    new[tile_index] |= in[index] != out[index];
  }
}

// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
// This is a ssandPile-specific version (generic version is defined in common.cl)
__kernel void ssandPile_update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  int2 pos = (int2)(x, y);
  unsigned c = cur [y * DIM + x];
  unsigned r = 0, v = 0, b = 0;

  if (c == 1)
    v = 255;
  else if (c == 2)
    b = 255;
  else if (c == 3)
    r = 255;
  else if (c == 4)
    r = v = b = 255;
  else if (c > 4)
    r = v = b = (2 * c);

  c = rgba(r, v, b, 0xFF);
  write_imagef (tex, pos, color_scatter (c));
}
