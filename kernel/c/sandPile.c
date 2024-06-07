#include <omp.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <unistd.h>

#include "easypap.h"
#include "hooks.h"


#ifdef ENABLE_VECTO
#include <immintrin.h>


#if __AVX512F__ == 1

typedef unsigned int TYPE;

static TYPE *restrict TABLE = NULL;
// Threashold = 10%
#define THRESHOLD 10

static unsigned cpu_y_part;
static unsigned gpu_y_part;
static cl_mem new_buff;
static TYPE * restrict TABLE_BUFF = NULL;
static unsigned iteration = 0;

static inline TYPE *
atable_cell (TYPE *restrict i, int y, int x)
{
  return i + y * DIM + x;
}

#define atable(y, x) (*atable_cell (TABLE, (y), (x)))

static inline TYPE *
table_cell (TYPE *restrict i, int step, int y, int x)
{
  return DIM * DIM * step + i + y * DIM + x;
}


#define AVX512_VEC_SIZE_INT 16
int tab[AVX512_VEC_SIZE_INT] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
__m512i vect ;

int ***tab_marker = NULL;

#define table(step, y, x) (*table_cell (TABLE, (step), (y), (x)))

static int in = 0;
static int out = 1;

static int current = 0;
static int next = 1;

static inline void
swap_tables ()
{
  int tmp = in;
  in = out;
  out = tmp;
}
static inline void
swap_tables_lazy ()
{
  int tmp = current;
  current = next;
  next = tmp;
}

#define RGB(r, g, b) rgba (r, g, b, 0xFF)

static TYPE max_grains;

void
asandPile_refresh_img ()
{
  unsigned long int max = 0;
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      {
        int g = table (in, i, j);
        int r, v, b;
        r = v = b = 0;
        if (g == 1)
          v = 255;
        else if (g == 2)
          b = 255;
        else if (g == 3)
          r = 255;
        else if (g == 4)
          r = v = b = 255;
        else if (g > 4)
          r = b = 255 - (240 * ((double)g) / (double)max_grains);

        cur_img (i, j) = RGB (r, v, b);
        if (g > max)
          max = g;
      }
  max_grains = max;
}


/////////////////////////////  Initial Configurations

static inline void set_cell (int y, int x, unsigned v)
{
  atable (y, x) = v;
  if (opencl_used)
    cur_img (y, x) = v;
}


void asandPile_draw_4partout (void);

void
asandPile_draw (char *param)
{
  // Call function ${kernel}_draw_${param}, or default function (second
  // parameter) if symbol not found
  hooks_draw_helper (param, asandPile_draw_4partout);
}

void
ssandPile_draw (char *param)
{
  hooks_draw_helper (param, asandPile_draw_4partout);
}

void
asandPile_draw_4partout (void)
{
  max_grains = 8;
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      set_cell(i, j, 4);
}

void
asandPile_draw_DIM (void)
{
  max_grains = DIM;
  for (int i = DIM / 4; i < DIM - 1; i += DIM / 4)
    for (int j = DIM / 4; j < DIM - 1; j += DIM / 4)
      set_cell (i, j, i * j / 4);
}

void
asandPile_draw_alea (void)
{
  max_grains = 5000;
  for (int i = 0; i<DIM>> 3; i++)
    {
      set_cell (1 + random() % (DIM - 2), 1 + random() % (DIM - 2), 1000 + (random() % (4000)));
    }
}

void
asandPile_draw_big (void)
{
  const int i = DIM / 2;
  set_cell (i, i, 100000);
}

static void
one_spiral (int x, int y, int step, int turns)
{
  int i = x, j = y, t;

  for (t = 1; t <= turns; t++)
    {
      for (; i < x + t * step; i++)
        set_cell (i, j, 3);
      for (; j < y + t * step + 1; j++)
        set_cell (i, j, 3);
      for (; i > x - t * step - 1; i--)
        set_cell (i, j, 3);
      for (; j > y - t * step - 1; j--)
        set_cell (i, j, 3);
    }
  set_cell (i, j, 4);

  for (int i = -2; i < 3; i++)
    for (int j = -2; j < 3; j++)
      set_cell (i + x, j + y, 3);
}

static void
many_spirals (int xdebut, int xfin, int ydebut, int yfin, int step, int turns)
{
  int i, j;
  int size = turns * step + 2;

  for (i = xdebut + size; i < xfin - size; i += 2 * size)
    for (j = ydebut + size; j < yfin - size; j += 2 * size)
      one_spiral (i, j, step, turns);
}

static void
spiral (unsigned twists)
{
  many_spirals (1, DIM - 2, 1, DIM - 2, 2, twists);
}

void
asandPile_draw_spirals (void)
{
  spiral (DIM / 32);
}

// shared functions

#define ALIAS(fun)                                                            \
  void ssandPile_##fun () { asandPile_##fun (); }

ALIAS (refresh_img);
ALIAS (draw_4partout);
ALIAS (draw_DIM);
ALIAS (draw_alea);
ALIAS (draw_big);
ALIAS (draw_spirals);


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Synchronous Kernel
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

#pragma GCC push_options
#pragma GCC optimize("unroll-all-loops")

void
ssandPile_init ()
{
  vect  = _mm512_loadu_epi32(&tab);
  TABLE = calloc (2 * DIM * DIM, sizeof (TYPE));
  if (tab_marker == NULL)
    {
      int tile_per_H = (DIM / TILE_H) +2;
      int tile_per_W = (DIM / TILE_W)+2;

      tab_marker = (int ***)malloc (2 * sizeof (int **));

      for (int i = 0; i < 2; i++)
        tab_marker[i] = (int **)malloc (tile_per_H * sizeof (int *));

      for (int j = 0; j < 2; j++)
        for (int i = 0; i < tile_per_H; i++)
          tab_marker[j][i] = (int *)malloc (tile_per_W * sizeof (int));

      for (int i = 0; i < tile_per_H; i++)
        for (int j = 0; j < tile_per_W; j++)
          {
            tab_marker[current][i][j] = 1;
            tab_marker[next][i][j] = 0;
          }
    }
}

void
ssandPile_finalize ()
{
  int tile_per_H = DIM / TILE_H + 2;
  for (int i = 0; i < tile_per_H; i++)
    {
      free (tab_marker[0][i]);
      free (tab_marker[1][i]);
    }
  free (tab_marker[0]);
  free (tab_marker[1]);
  free (tab_marker);

  free (TABLE);
  free (TABLE_BUFF);
}



int
ssandPile_do_tile_default (int x, int y, int width, int height)
{
  int diff = 0;
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      {
        table (out, i, j) = table (in, i, j) % 4;
        table (out, i, j) += table (in, i + 1, j) / 4;
        table (out, i, j) += table (in, i - 1, j) / 4;
        table (out, i, j) += table (in, i, j + 1) / 4;
        table (out, i, j) += table (in, i, j - 1) / 4;
        if (table (out, i, j) >= 4)
          diff = 1;
      }
  return diff;
}
// ssandPile_do_tile_opt
int
ssandPile_do_tile_opt (int x, int y, int width, int height)
{
  int diff = 0;
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      {
        table (out, i, j) = table (in, i, j) % 4 + table (in, i + 1, j) / 4
                            + table (in, i - 1, j) / 4
                            + table (in, i, j + 1) / 4
                            + table (in, i, j - 1) / 4;
        if (table (out, i, j) >= 4)
          diff = 1;
      }
  return diff;
}

int
ssandPile_do_tile_lazy (int x, int y, int width, int height)
{
  int diff = 0;
  int x1 = x / TILE_W+1;
  int y1 = y / TILE_H+1;
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      {
        table (out, i, j) = table (in, i, j) % 4 + table (in, i + 1, j) / 4
                            + table (in, i - 1, j) / 4
                            + table (in, i, j + 1) / 4
                            + table (in, i, j - 1) / 4;
        if (table (out, i, j) != table (in, i, j))
          {
            if (i == y )
              tab_marker[next][y1 - 1][x1] = 1;

            if (i == (y + height - 1) )
              tab_marker[next][y1 + 1][x1] = 1;

            if (j == x )
              tab_marker[next][y1][x1 - 1] = 1;

            if (j == (x + width - 1))
              tab_marker[next][y1][x1 + 1] = 1;
            diff = 1;
          }
      }
  if (diff == 1)
    tab_marker[next][y1][x1] = 1;
  return diff;
}

int
ssandPile_do_tile_avx (int x, int y, int width, int height)
{

  int diff = 0;
  __m512i three = _mm512_set1_epi32 (3);
  __m512i line_vec,line,lineup_vec,linedw_vec,lineR_vec,lineL_vec,outside,diffLine;

  for (int i = y; i < y + height; i++)
    {
      for (int j = x; j < x + width; j += AVX512_VEC_SIZE_INT)
      {
        line = _mm512_loadu_epi32(&table(in,i,j)) ;
        line_vec = _mm512_and_si512 (line, three);

        lineup_vec = _mm512_loadu_epi32(&table(in,i-1,j));
        lineup_vec = _mm512_srli_epi32 (lineup_vec, 2);

        linedw_vec = _mm512_loadu_epi32(&table(in,i+1,j));
        linedw_vec = _mm512_srli_epi32 (linedw_vec, 2);

        lineR_vec = _mm512_loadu_epi32(&table(in,i,j+1));
        lineR_vec = _mm512_srli_epi32 (lineR_vec, 2);

        lineL_vec = _mm512_loadu_epi32(&table(in,i,j-1));
        lineL_vec = _mm512_srli_epi32 (lineL_vec, 2);

        line_vec = _mm512_add_epi32 (line_vec,_mm512_add_epi32 (lineup_vec,
                  _mm512_add_epi32 (_mm512_add_epi32 (lineL_vec, lineR_vec),
                                    linedw_vec)));
        outside = _mm512_set1_epi32(x + width);
        __mmask16 mask = _mm512_cmpgt_epi32_mask(outside,_mm512_add_epi32(vect,_mm512_set1_epi32(j)));
        _mm512_mask_storeu_epi32(&table(out,i,j),mask,line_vec);

        diffLine = _mm512_sub_epi32(line,line_vec);
        int a = _mm512_mask_reduce_add_epi32(mask,_mm512_abs_epi32(diffLine));

        if(a!=0)
          diff=1;
      }
    }  
  return diff;
}
int
ssandPile_do_tile_lazy_avx (int x, int y, int width, int height)
{
  int x1 = x / TILE_W+1;
  int y1 = y / TILE_H+1;
  int diff = 0;
  __m512i three = _mm512_set1_epi32 (3);
  __m512i line_vec,line,lineup_vec,linedw_vec,lineR_vec,lineL_vec,outside,diffLine;

  for (int i = y; i < y + height; i++)
    {
      for (int j = x; j < x + width; j += AVX512_VEC_SIZE_INT)
      {
        line = _mm512_loadu_epi32(&table(in,i,j)) ;
        line_vec = _mm512_and_si512 (line, three);

        lineup_vec = _mm512_loadu_epi32(&table(in,i-1,j));
        lineup_vec = _mm512_srli_epi32 (lineup_vec, 2);

        linedw_vec = _mm512_loadu_epi32(&table(in,i+1,j));
        linedw_vec = _mm512_srli_epi32 (linedw_vec, 2);

        lineR_vec = _mm512_loadu_epi32(&table(in,i,j+1));
        lineR_vec = _mm512_srli_epi32 (lineR_vec, 2);

        lineL_vec = _mm512_loadu_epi32(&table(in,i,j-1));
        lineL_vec = _mm512_srli_epi32 (lineL_vec, 2);

        line_vec = _mm512_add_epi32 (line_vec,_mm512_add_epi32 (lineup_vec,
                  _mm512_add_epi32 (_mm512_add_epi32 (lineL_vec, lineR_vec),
                                    linedw_vec)));
        outside = _mm512_set1_epi32(x + width);
        __mmask16 mask = _mm512_cmpgt_epi32_mask(outside,_mm512_add_epi32(vect,_mm512_set1_epi32(j)));
        _mm512_mask_storeu_epi32(&table(out,i,j),mask,line_vec);

        diffLine = _mm512_sub_epi32(line,line_vec);
        int a = _mm512_mask_reduce_add_epi32(mask,_mm512_abs_epi32(diffLine));

        if(a!=0){
          if (i == y )
              tab_marker[next][y1 - 1][x1] = 1;
          if (i == (y + height - 1) )
              tab_marker[next][y1 + 1][x1] = 1;
          if (table (out, i, j) != table (in, i, j) && j==x )
              tab_marker[next][y1][x1 - 1] = 1;
          if (table (out, i, x + width-1) != table (in, i, x + width-1) )
              tab_marker[next][y1][x1 + 1] = 1;
          diff=1;
        } 
      }
    }
  if(diff==1)
      tab_marker[next][y1][x1] = 1;  
  return diff;
}
/////////////////////////////////////////////////////////////////////////  
////////////////////// compute //////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


// Renvoie le nombre d'itÃ©rations effectuÃ©es avant stabilisation, ou 0
unsigned
ssandPile_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = do_tile (1, 1, DIM - 2, DIM - 2, 0);
      swap_tables ();
      if (change == 0)
        return it;
    }
  return 0;
}

unsigned
ssandPile_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = 0;
      for (int y = 0; y < DIM; y += TILE_H)
        for (int x = 0; x < DIM; x += TILE_W)
          change |= do_tile (x + (x == 0), y + (y == 0),
                             TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                             TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                             0 /* CPU id */);
      swap_tables ();
      if (change == 0)
        return it;
    }
  return 0;
}


unsigned
ssandPile_compute_lazy (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = 0;
      for (int y = 0; y < DIM; y += TILE_H)
        for (int x = 0; x < DIM; x += TILE_W)
          {
            int y1 = (y / TILE_H)+1;
            int x1 = (x / TILE_W)+1;
            if (tab_marker[current][y1][x1] == 1)
              {
                change |= do_tile (x + (x == 0), y + (y == 0),
                               TILE_W - (((x + TILE_W) == DIM) + (x == 0)),
                               TILE_H - (((y + TILE_H) == DIM) + (y == 0)),
                               omp_get_thread_num ());

                tab_marker[current][y1][x1] = 0;
              }
          }
      swap_tables_lazy ();
      swap_tables ();
      if (change == 0)
        return it;
    }
  return 0;
}
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// OMP synchronous Kernel
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

unsigned
ssandPile_compute_omp (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int diff = 0;
#pragma omp parallel for schedule(runtime)
      for (int i = 1; i < DIM - 1; i++)
        for (int j = 1; j < DIM - 1; j++)
          {
            table (out, i, j) = table (in, i, j) % 4 + table (in, i + 1, j) / 4
                                + table (in, i - 1, j) / 4
                                + table (in, i, j + 1) / 4
                                + table (in, i, j - 1) / 4;
            if (table (out, i, j) >= 4)
              diff = 1;
          }
      int change = diff;
      swap_tables ();
      if (change == 0)
        return it;
    }
  return 0;
}

unsigned
ssandPile_compute_omp_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = 0;
#pragma omp parallel for collapse(2) shared(DIM, TILE_W, TILE_H) schedule(runtime) reduction(|: change)
      for (int y = 0; y < DIM; y += TILE_H)
        for (int x = 0; x < DIM; x += TILE_W)
          {
            change |= do_tile (x + (x == 0), y + (y == 0),
                                TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                omp_get_thread_num () /* CPU id */);
          }
      swap_tables ();
      if (change == 0)
        return it;
    }
  return 0;
}

unsigned
ssandPile_compute_omp_taskloop (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {

      int change = 0;
#pragma omp parallel
#pragma omp single
#pragma omp taskloop collapse(2) shared(DIM, TILE_W, TILE_H) reduction (|: change) 
      for (int y = 0; y < DIM; y += TILE_H)
        for (int x = 0; x < DIM; x += TILE_W)
          {
            change |= do_tile (x + (x == 0), y + (y == 0),
                                TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                omp_get_thread_num () /* CPU id */);
          }
      swap_tables ();
      if (change == 0)
        return it;
    }

  return 0;
}

unsigned
ssandPile_compute_omp_lazy (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = 0;
#pragma omp parallel for collapse(2) schedule(runtime) reduction(|: change)
      for (int y = 0; y < DIM; y += TILE_H)
        for (int x = 0; x < DIM; x += TILE_W)
          {
            int y1 = y / TILE_H+1;
            int x1 = x / TILE_W+1;
            if (tab_marker[current][y1][x1] == 1)
              {
                change |= do_tile (x + (x == 0), y + (y == 0),
                               TILE_W - (((x + TILE_W) == DIM) + (x == 0)),
                               TILE_H - (((y + TILE_H) == DIM) + (y == 0)),
                               omp_get_thread_num ());
                tab_marker[current][y1][x1] = 0;
              }
          }
      swap_tables_lazy ();
      swap_tables ();
      if (change == 0)
        return it;
    }
  return 0;
}

// Only called when --dump or --thumbnails is used
void ssandPile_refresh_img_ocl ()
{
  cl_int err;

  err =
      clEnqueueReadBuffer (queue, cur_buffer, CL_TRUE, 0,
                           sizeof (unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
  check (err, "Failed to read buffer from GPU");

  ssandPile_refresh_img ();
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Asynchronous Kernel
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void
asandPile_init ()
{
  in = out = 0;
  vect  = _mm512_loadu_epi32(&tab);
  if (TABLE == NULL)
    {
      const unsigned size = DIM * DIM * sizeof (TYPE);

      PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

      TABLE = mmap (NULL, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
  if (tab_marker == NULL)
    {
      int tile_per_H = DIM / TILE_H+2;
      int tile_per_W = DIM / TILE_W+2;

      tab_marker = (int ***)malloc (2 * sizeof (int **));

      for (int i = 0; i < 2; i++)
        tab_marker[i] = (int **)malloc (tile_per_H * sizeof (int *));

      for (int j = 0; j < 2; j++)
        for (int i = 0; i < tile_per_H; i++)
          tab_marker[j][i] = (int *)malloc (tile_per_W * sizeof (int));

      for (int i = 0; i < tile_per_H; i++)
        for (int j = 0; j < tile_per_W; j++)
          {
            tab_marker[current][i][j] = 1;
            tab_marker[next][i][j] = 0;
          }
    }
}

void
asandPile_finalize ()
{
  const unsigned size = DIM * DIM * sizeof (TYPE);
  for (int i = 0; i < ((DIM / TILE_H)+2); i++)
    {
      free (tab_marker[0][i]);
      free (tab_marker[1][i]);
    }
  free (tab_marker[0]);
  free (tab_marker[1]);
  free (tab_marker);

  munmap (TABLE, size);
}

int
asandPile_do_tile_default (int x, int y, int width, int height)
{
  int change = 0;
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      if (atable (i, j) >= 4)
        {
          atable (i, j - 1) += atable (i, j) / 4;
          atable (i, j + 1) += atable (i, j) / 4;
          atable (i - 1, j) += atable (i, j) / 4;
          atable (i + 1, j) += atable (i, j) / 4;
          atable (i, j) %= 4;
          change = 1;
        }
  return change;
}

int
asandPile_do_tile_opt (int x, int y, int width, int height)
{
  int change = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      if (atable (i, j) >= 4)
        {
          TYPE div_res = atable (i, j) / 4;
          atable (i, j) %= 4;
          atable (i, j - 1) += div_res;
          atable (i, j + 1) += div_res;
          atable (i - 1, j) += div_res;
          atable (i + 1, j) += div_res;
          change = 1;
        }
  return change;
}
int
asandPile_do_tile_avx (int x, int y, int width, int height)
{
  int change = 0;
  __m512i zero = _mm512_set1_epi32(0);
  __m512i three = _mm512_set1_epi32(3);

  __m512i line,lineLeft,lineRight,mod,line_vec,d,line_up_vec,line_dw_vec,diffLine;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j+=AVX512_VEC_SIZE_INT)
    {
        __mmask16 mask = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(x + width),_mm512_add_epi32(vect,_mm512_set1_epi32(j)));

        line = _mm512_loadu_epi32(&atable(i,j));
        d = _mm512_srli_epi32(line,2);

        lineLeft  = _mm512_alignr_epi32(zero,_mm512_mask_blend_epi32(mask,zero,d),1);//_mm512_alignr_epi32(zero,d,1);
        lineRight = _mm512_alignr_epi32(d,zero,AVX512_VEC_SIZE_INT - 1);

        mod = _mm512_and_si512(line,three);
        line_vec = _mm512_add_epi32(mod,_mm512_add_epi32(lineRight,lineLeft));

        line_up_vec = _mm512_loadu_epi32(&atable(i-1,j));
        line_up_vec = _mm512_add_epi32(line_up_vec,d);

        line_dw_vec = _mm512_loadu_epi32(&atable(i+1,j));
        line_dw_vec = _mm512_add_epi32(line_dw_vec,d);

        int indice = _mm512_reduce_add_epi32(_mm512_mask_blend_epi32(mask,zero,three))/3;
        
        _mm512_mask_storeu_epi32(&atable(i,j),mask,line_vec);
        _mm512_mask_storeu_epi32(&atable(i-1,j),mask,line_up_vec);
        _mm512_mask_storeu_epi32(&atable(i+1,j),mask,line_dw_vec);

        int blended[AVX512_VEC_SIZE_INT] ;
        _mm512_storeu_epi32(&blended,d);

        atable(i,j-1) = atable(i,j-1) + blended[0];
        atable(i,j+indice) = atable(i,j+indice) + blended[indice-1];

        diffLine = _mm512_mask_sub_epi32(zero,mask,line,line_vec);
        int a = _mm512_reduce_add_epi32(_mm512_abs_epi32(diffLine));
        if(a!=0)
          change=1;
      
    }
      
  return change;
}
int
asandPile_do_tile_lazy_avx (int x, int y, int width, int height)
{
  int change = 0;
  int x1 = x / TILE_W + 1;
  int y1 = y / TILE_H + 1;
  __m512i zero = _mm512_set1_epi32(0);
  __m512i three = _mm512_set1_epi32(3);

  __m512i line,lineLeft,lineRight,mod,line_vec,d,line_up_vec,line_dw_vec,diffLine;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j+=AVX512_VEC_SIZE_INT)
    {
        if (atable (i, j) >=  4  && j==x )
              tab_marker[next][y1][x1 - 1] = 1;
        if (atable(i, x + width-1) >= 4  )
              tab_marker[next][y1][x1 + 1] = 1;
        __mmask16 mask = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(x + width),_mm512_add_epi32(vect,_mm512_set1_epi32(j)));

        line = _mm512_loadu_epi32(&atable(i,j));
        d = _mm512_srli_epi32(line,2);

        lineLeft  = _mm512_alignr_epi32(zero,_mm512_mask_blend_epi32(mask,zero,d),1);//_mm512_alignr_epi32(zero,d,1);
        lineRight = _mm512_alignr_epi32(d,zero,AVX512_VEC_SIZE_INT - 1);

        mod = _mm512_and_si512(line,three);
        line_vec = _mm512_add_epi32(mod,_mm512_add_epi32(lineRight,lineLeft));

        line_up_vec = _mm512_loadu_epi32(&atable(i-1,j));
        line_up_vec = _mm512_add_epi32(line_up_vec,d);

        line_dw_vec = _mm512_loadu_epi32(&atable(i+1,j));
        line_dw_vec = _mm512_add_epi32(line_dw_vec,d);

        int indice = _mm512_reduce_add_epi32(_mm512_mask_blend_epi32(mask,zero,three))/3;
        
        _mm512_mask_storeu_epi32(&atable(i,j),mask,line_vec);
        _mm512_mask_storeu_epi32(&atable(i-1,j),mask,line_up_vec);
        _mm512_mask_storeu_epi32(&atable(i+1,j),mask,line_dw_vec);

        int blended[AVX512_VEC_SIZE_INT] ;
        _mm512_storeu_epi32(&blended,d);

        atable(i,j-1) = atable(i,j-1) + blended[0];
        atable(i,j+indice) = atable(i,j+indice) + blended[indice-1];

        diffLine = _mm512_mask_sub_epi32(zero,mask,line,line_vec);
        int a = _mm512_reduce_add_epi32(_mm512_abs_epi32(diffLine));
        if(a!=0){
          if (i == y )
              tab_marker[next][y1 - 1][x1] = 1;
          if (i == (y + height - 1) )
              tab_marker[next][y1 + 1][x1] = 1;
          change=1;
        } 
      
    }
  if(change==1)
      tab_marker[next][y1][x1] = 1;
  return change;
}

int
asandPile_do_tile_omp (int x, int y, int width, int height)
{
  int change = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      if ((i == y && (j == x || j == (x + width - 1)))
          || (i == (y + height - 1) && (j == x || j == (x + width - 1))))
        {
#pragma omp critical
          if (atable (i, j) >= 4)
            {
              TYPE div_res = atable (i, j) / 4;
              atable (i, j) %= 4;
              atable (i, j - 1) += div_res;
              atable (i, j + 1) += div_res;
              atable (i - 1, j) += div_res;
              atable (i + 1, j) += div_res;
              change = 1;
            }
        }
      else if (atable (i, j) >= 4)
        {
          TYPE div_res = atable (i, j) / 4;
          atable (i, j) %= 4;
          atable (i, j - 1) += div_res;
          atable (i, j + 1) += div_res;
          atable (i - 1, j) += div_res;
          atable (i + 1, j) += div_res;
          change = 1;
        }
  return change;
}

int
asandPile_do_tile_lazy (int x, int y, int width, int height)
{
  int diff = 0;
  int x1 = x / TILE_W + 1;
  int y1 = y / TILE_H + 1;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      {
        if ((i == y && (j == x || j == (x + width - 1)))
            || (i == (y + height - 1) && (j == x || j == (x + width - 1))))
          {
#pragma omp critical
            if (atable (i, j) >= 4)
              {
                TYPE div_res = atable (i, j) / 4;
                atable (i, j) %= 4;
                atable (i, j - 1) += div_res;
                atable (i, j + 1) += div_res;
                atable (i - 1, j) += div_res;
                atable (i + 1, j) += div_res;

                if (i == y )
                  tab_marker[next][y1 - 1][x1] = 1;

                if (i == (y + height - 1) )
                  tab_marker[next][y1 + 1][x1] = 1;

                if (j == x && j != 1)
                  tab_marker[next][y1][x1 - 1] = 1;

                if (j == (x + width - 1) )
                  tab_marker[next][y1][x1 + 1] = 1;
                diff = 1;
              }
          }
        else
          {
            if (atable (i, j) >= 4)
              {
                TYPE div_res = atable (i, j) / 4;
                atable (i, j) %= 4;
                atable (i, j - 1) += div_res;
                atable (i, j + 1) += div_res;
                atable (i - 1, j) += div_res;
                atable (i + 1, j) += div_res;

                if (i == y )
                  tab_marker[next][y1 - 1][x1] = 1;

                if (i == (y + height - 1) )
                  tab_marker[next][y1 + 1][x1] = 1;

                if (j == x )
                  tab_marker[next][y1][x1 - 1] = 1;

                if (j == (x + width - 1) )
                  tab_marker[next][y1][x1 + 1] = 1;
                diff = 1;
              }
          }
      }

  if (diff == 1)
    tab_marker[next][y1][x1] = 1;
  return diff;
}

int
asandPile_do_tile_lazy_alt (int x, int y, int width, int height)
{
  int diff = 0;
  int x1 = x / TILE_W+1;
  int y1 = y / TILE_H+1;
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      {
        if (atable (i, j) >= 4)
          {
            TYPE div_res = atable (i, j) / 4;
            atable (i, j) %= 4;
            atable (i, j - 1) += div_res;
            atable (i, j + 1) += div_res;
            atable (i - 1, j) += div_res;
            atable (i + 1, j) += div_res;
            if (i == y )
              tab_marker[next][y1 - 1][x1] = 1;

            if (i == (y + height - 1) )
              tab_marker[next][y1 + 1][x1] = 1;

            if (j == x )
              tab_marker[next][y1][x1 - 1] = 1;

            if (j == (x + width - 1) )
              tab_marker[next][y1][x1 + 1] = 1;

            diff = 1;
          }
      }

  if (diff == 1)
    tab_marker[next][y1][x1] = 1;
  return diff;
}
///////////////////////////////////////////////////////////////////////////
/////////////////////////////// End do_tile ///////////////////////////////
///////////////////////////////////////////////////////////////////////////

unsigned
asandPile_compute_seq (unsigned nb_iter)
{
  int change = 0;
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      // On traite toute l'image en un coup (oui, c'est une grosse tuile)
      change = do_tile (1, 1, DIM - 2, DIM - 2, 0);

      if (change == 0)
        return it;
    }
  return 0;
}

unsigned
asandPile_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = 0;

      for (int y = 0; y < DIM; y += TILE_H)
        for (int x = 0; x < DIM; x += TILE_W)
          change |= do_tile (x + (x == 0), y + (y == 0),
                             TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                             TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                             0 /* CPU id */);
      if (change == 0)
        return it;
    }

  return 0;
}

unsigned
asandPile_compute_lazy (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = 0;
      for (int w = 0; w < 2; w++)
        {
          for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W)
              {
                int test = (x / TILE_W % 2) == (y / TILE_H % 2);
                if (w == test)
                  {
                    int y1 = y / TILE_H+1;
                    int x1 = x / TILE_W+1;
                    if (tab_marker[current][y1][x1] == 1)
                      {
                        change |= do_tile (
                            x + (x == 0), y + (y == 0),
                            TILE_W - (((x + TILE_W) == DIM) + (x == 0)),
                            TILE_H - (((y + TILE_H) == DIM) + (y == 0)),
                            omp_get_thread_num ());

                        tab_marker[current][y1][x1] = 0;
                      }
                  }
              }
        }
      swap_tables_lazy ();
      if (change == 0)
        return it;
    }
  return 0;
}

unsigned
asandPile_compute_lazy_alt (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = 0;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          for (int y = i * TILE_H; y < DIM; y += 3 * TILE_H)
            for (int x = j * TILE_W; x < DIM; x += 3 * TILE_W)
              {
                int y1 = y / TILE_H+1;
                int x1 = x / TILE_W+1;
                if (tab_marker[current][y1][x1] == 1)
                  {
                    change |= do_tile (
                        x + (x == 0), y + (y == 0),
                        TILE_W - (((x + TILE_W) == DIM) + (x == 0)),
                        TILE_H - (((y + TILE_H) == DIM) + (y == 0)),
                        omp_get_thread_num ());

                    tab_marker[current][y1][x1] = 0;
                  }
              }
      swap_tables_lazy ();
      if (change == 0)
        return it;
    }
  return 0;
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// OMP Asynchronous Kernel
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

unsigned
asandPile_compute_omp (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = 0;
      for (unsigned perm_num = 1; perm_num < 4;
           perm_num++) // ligne number is valid each three lignes to avoid rw
                       // conflict | every ligne is treated seperately
#pragma omp parallel for schedule(runtime)
        for (int i = perm_num; i < DIM - 1; i += 3)
          for (int j = 1; j < DIM - 1; j++)
            {
              if (atable (i, j) >= 4)
                {
                  TYPE div_res = atable (i, j) / 4;
                  atable (i, j) %= 4;
                  atable (i, j - 1) += div_res;
                  atable (i, j + 1) += div_res;
                  atable (i - 1, j) += div_res;
                  atable (i + 1, j) += div_res;
                  change = 1;
                }
            }
      if (change == 0)
        return it;
    }
  return 0;
}

// assand  pile with tile to be called with -wt do tile opt
unsigned
asandPile_compute_omp_tiled_alt (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = 0;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
#pragma omp parallel for collapse(2) schedule(runtime) reduction(|: change)
          for (int y = i * TILE_H; y < DIM; y += 3 * TILE_H)
            for (int x = j * TILE_W; x < DIM; x += 3 * TILE_W)
              {
                change |= do_tile (x + (x == 0), y + (y == 0),
                                    TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                    TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                    omp_get_thread_num ());
              }
      if (change == 0)
        return it;
    }
  return 0;
}

// assand  pile with tile to be called with -wt do tile omp
unsigned
asandPile_compute_omp_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = 0;
      for (int w = 0; w < 2; w++)
        {
#pragma omp parallel for collapse(2) schedule(runtime) reduction(|: change)
          for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W)
              {
                int ok = (x / TILE_W % 2) == (y / TILE_H % 2);
                if (w == ok)
                  {
                    change |= do_tile (x + (x == 0), y + (y == 0),
                                   TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                                   TILE_H - ((y + TILE_H == DIM) + (y == 0)),
                                   omp_get_thread_num ());
                  }
              }
        }
      if (change == 0)
        return it;
    }
  return 0;
}

unsigned
asandPile_compute_omp_lazy (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = 0;
      for (int w = 0; w < 2; w++)
        {
#pragma omp parallel for collapse(2) schedule(runtime) reduction(|: change)
          for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W)
              {
                int test = (x / TILE_W % 2) == (y / TILE_H % 2);
                if (w == test)
                  {
                    int y1 = y / TILE_H+1;
                    int x1 = x / TILE_W+1;
                    if (tab_marker[current][y1][x1] == 1)
                      {
                        change |= do_tile (
                            x + (x == 0), y + (y == 0),
                            TILE_W - (((x + TILE_W) == DIM) + (x == 0)),
                            TILE_H - (((y + TILE_H) == DIM) + (y == 0)),
                            omp_get_thread_num ());

                        tab_marker[current][y1][x1] = 0;
                      }
                  }
              }
        }
      swap_tables_lazy ();
      if (change == 0)
        return it;
    }
  return 0;
}

unsigned
asandPile_compute_omp_lazy_alt (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    {
      int change = 0;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
#pragma omp parallel for collapse(2) schedule(runtime) reduction(|: change)
          for (int y = i * TILE_H; y < DIM; y += 3 * TILE_H)
            for (int x = j * TILE_W; x < DIM; x += 3 * TILE_W)
              {
                int y1 = y / TILE_H+1;
                int x1 = x / TILE_W+1;
                if (tab_marker[current][y1][x1] == 1)
                  {
                    change |= do_tile (x + (x == 0), y + (y == 0),
                                   TILE_W - (((x + TILE_W) == DIM) + (x == 0)),
                                   TILE_H - (((y + TILE_H) == DIM) + (y == 0)),
                                   omp_get_thread_num ());
                    tab_marker[current][y1][x1] = 0;
                  }
              }
      swap_tables_lazy ();
      if (change == 0)
        return it;
    }
  return 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////OCL-HYBRID//////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static long gpu_duration = 0, cpu_duration = 0;

static int much_greater_than (long t1, long t2)
{
  return (t1 > t2) && ((t1 - t2) * 100 / t1 > THRESHOLD);
}

void ssandPile_refresh_img_ocl_hybrid ()
{
  cl_int err;

  err =
      clEnqueueReadBuffer (queue, cur_buffer, CL_TRUE, cpu_y_part * DIM * sizeof(unsigned)  ,
                           sizeof (unsigned) * DIM * gpu_y_part, &table(in, (gpu_y_part), 0), 0, NULL, NULL);
  check (err, "Failed to read buffer from GPU");

  ssandPile_refresh_img ();
}

void ssandPile_init_ocl_hybrid (void)
{
  ssandPile_init();

  if (GPU_TILE_H != TILE_H)
    exit_with_error ("CPU and GPU Tiles should have the same height (%d != %d)",
                     GPU_TILE_H, TILE_H);

  cpu_y_part = (NB_TILES_Y / 2) * GPU_TILE_H; // Start with fifty-fifty
  gpu_y_part = DIM - cpu_y_part;

  size_t size = (DIM / GPU_TILE_W) * (gpu_y_part / GPU_TILE_H);
  TABLE_BUFF = (TYPE *) malloc(size * sizeof(TYPE));

  if (!TABLE_BUFF)
    exit_with_error ("unable to allocate new buffer table!!\n");

  new_buff = clCreateBuffer (context, CL_MEM_READ_WRITE,
                               sizeof (TYPE) * size , NULL, NULL); // to check size 

  if (!new_buff)
    exit_with_error ("unable to allocate new buffer!!\n");
}

unsigned ssandPile_invoke_ocl_hybrid (unsigned nb_iter)
{
  size_t global[2] = {DIM,
                      gpu_y_part}; // global domain size for our calculation
  size_t local[2]  = {GPU_TILE_W,
                     GPU_TILE_H}; // local domain size for our calculation
  cl_int err;
  cl_event kernel_event;
  long t1, t2;
  int gpu_accumulated_lines = 0;
  int diff = 0;

  // Load balancing
  // if (gpu_duration != 0) {
    // if (much_greater_than (gpu_duration, cpu_duration) &&
    //     gpu_y_part > GPU_TILE_H) {
    //   gpu_y_part -= GPU_TILE_H;
    //   cpu_y_part += GPU_TILE_H;
    //   global[1] = gpu_y_part;
    // } 
    // else
  //   if (much_greater_than (cpu_duration, gpu_duration) &&
  //    cpu_y_part > GPU_TILE_H) {
  //     gpu_y_part += GPU_TILE_H;
  //     cpu_y_part -= GPU_TILE_H;
  //     global[1] = gpu_y_part;
  //   }
  // }
  size_t size = (DIM / GPU_TILE_W) * (gpu_y_part / GPU_TILE_H);
  // TABLE_BUFF = realloc(TABLE_BUFF, size * sizeof(TYPE));
  // cl_mem new = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof (TYPE) * size , NULL, NULL);

  for (unsigned it = 1; it <= nb_iter; it++) {
    ++iteration;
    // Set kernel arguments
    
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &next_buffer);
    err |= clSetKernelArg (compute_kernel, 2, sizeof (cl_mem), &new_buff);
    err |= clSetKernelArg (compute_kernel, 3, sizeof (unsigned), &cpu_y_part);
    err |= clSetKernelArg (compute_kernel, 4, sizeof (unsigned), &iteration);

    check (err, "Failed to set kernel arguments");

    // Launch GPU kernel
    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, &kernel_event);
    check (err, "Failed to execute kernel");

    clFlush (queue);

    t1 = what_time_is_it ();
    int change = 0;
    #pragma omp parallel for collapse(2) schedule(runtime) reduction (|: change)
    for (int y = 0; y < cpu_y_part; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        change |=
            do_tile(x + (x == 0), y + (y == 0),
                    TILE_W - ((x + TILE_W == DIM) + (x == 0)),
                    TILE_H - ((y + TILE_H == DIM) + (y == 0)), omp_get_thread_num ());

    t2 = what_time_is_it ();

    //READ WRITE

    err = clEnqueueReadBuffer (queue, next_buffer, CL_TRUE, (cpu_y_part) * DIM * sizeof(unsigned) ,
                                DIM * sizeof (unsigned), &table(out, (cpu_y_part), 0), 0,
                                NULL, NULL);

    check (err, "Failed to read to buffer");

    err = clEnqueueWriteBuffer (queue, next_buffer, CL_TRUE, (cpu_y_part - 1) * DIM * sizeof(unsigned) ,
                                DIM * sizeof (unsigned), &table(out, (cpu_y_part - 1), 0), 0,
                                NULL, NULL);

    check (err, "Failed to write to buffer");

    
    // SWAP
    {
      cl_mem tmp  = cur_buffer;
      cur_buffer  = next_buffer;
      next_buffer = tmp;
    }

    swap_tables ();
  // printf("DIM = %d size = %d gpu = %d cpu = %d \n", DIM, size, gpu_y_part, cpu_y_part);

    if (change == 0 && iteration % (DIM/8) == 0){
      err = clEnqueueReadBuffer (queue, new_buff, CL_TRUE, 0, sizeof(unsigned) * size, TABLE_BUFF, 0, NULL, NULL);

      check (err, "Failed to read to buffer");

      #pragma omp parallel for schedule(runtime) reduction(|: change)
      for (int i = 0; i < size ; i++){
        change |= TABLE_BUFF[i];
      }

      if (!change){
        diff = it;
        break;
      }
    }
    cpu_duration = t2 - t1;
    gpu_duration = ocl_monitor (kernel_event, 0, cpu_y_part, global[0], global[1], TASK_TYPE_COMPUTE);
    clReleaseEvent (kernel_event);
    gpu_accumulated_lines += gpu_y_part;
    
    clFinish (queue);
                                
  }

  if (do_display) {
    // Send CPU contribution to GPU memory
    err = clEnqueueWriteBuffer (queue, cur_buffer, CL_TRUE, 0 ,
                                DIM * (cpu_y_part - 1) * sizeof (unsigned), &table(in, 0, 0), 0,
                                NULL, NULL);
    check (err, "Failed to write to buffer");
  } else
    PRINT_DEBUG ('u', "In average, GPU took %.1f%% of the lines\n",
                 (float)gpu_accumulated_lines * 100 / (DIM * nb_iter));

// free(TABLE_BUFF);

  return diff;
}

#pragma GCC pop_options
#endif
#endif