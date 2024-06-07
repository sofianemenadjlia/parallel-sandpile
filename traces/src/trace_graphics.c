#include <SDL_image.h>
#include <SDL_opengl.h>
#include <SDL_ttf.h>
#include <fcntl.h>
#include <fut.h>
#include <fxt-tools.h>
#include <fxt.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#include "error.h"
#include "trace_common.h"
#include "trace_graphics.h"

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

// How much percentage of duration should we shift ?
#define SHIFT_FACTOR 0.02
#define MIN_DURATION 100.0

#define WINDOW_MIN_WIDTH 1024
// no WINDOW_MIN_HEIGHT: needs to be automatically computed

#define MIN_TASK_HEIGHT 4
#define MAX_TASK_HEIGHT 44

#define MAX_PREVIEW_DIM 512
#define MIN_PREVIEW_DIM 256

#define Y_MARGIN 5
#define CPU_ROW_HEIGHT (TASK_HEIGHT + 2 * Y_MARGIN + 2)
#define GANTT_WIDTH (WINDOW_WIDTH - (LEFT_MARGIN + PREVIEW_DIM + TOP_MARGIN))
#define LEFT_MARGIN 64
#define TOP_MARGIN 48
#define FONT_HEIGHT 20
#define BOTTOM_MARGIN (FONT_HEIGHT + 4)
#define INTERTRACE_MARGIN (2 * FONT_HEIGHT + 2)

#define PRELOAD_THUMBNAILS 1

#define SQUARE_SIZE 16
#define TILE_ALPHA 0x80
#define BUTTON_ALPHA 60

static SDL_Texture **square_tex_bright = NULL;
static SDL_Texture **square_tex_dark   = NULL;
static SDL_Texture *black_square       = NULL;
static SDL_Texture *white_square       = NULL;

#ifdef PRELOAD_THUMBNAILS
static SDL_Texture **thumb_tex[MAX_TRACES] = {NULL, NULL};
#endif

static int TASK_HEIGHT   = MAX_TASK_HEIGHT;
static int WINDOW_HEIGHT = -1;
static int WINDOW_WIDTH  = -1;

static int GANTT_HEIGHT, PREVIEW_DIM;

static int BUBBLE_WIDTH, BUBBLE_HEIGHT;

static SDL_Window *window           = NULL;
static SDL_Renderer *renderer       = NULL;
static TTF_Font *the_font           = NULL;
static SDL_Texture **perf_fill      = NULL;
static SDL_Texture *text_texture    = NULL;
static SDL_Texture *vertical_line   = NULL;
static SDL_Texture *horizontal_line = NULL;
static SDL_Texture *horizontal_bis  = NULL;
static SDL_Texture *bulle_tex       = NULL;
static SDL_Texture *us_tex          = NULL;
static SDL_Texture *sigma_tex       = NULL;
static SDL_Texture *tab_left        = NULL;
static SDL_Texture *tab_right       = NULL;
static SDL_Texture *tab_high        = NULL;
static SDL_Texture *tab_low         = NULL;
static SDL_Texture *align_tex       = NULL;
static SDL_Texture *quick_nav_tex   = NULL;
static SDL_Texture *track_tex       = NULL;
static SDL_Texture *footprint_tex   = NULL;
static SDL_Texture *digit_tex[10]   = {NULL};

static SDL_Rect align_rect, quick_nav_rect, track_rect, footprint_rect;

static unsigned digit_tex_width[10];
static unsigned digit_tex_height;

static int quick_nav_mode = 0;
static int horiz_mode     = 0;
static int tracking_mode  = 0;
static int footprint_mode = 0;

static long start_time = 0, end_time = 0, duration = 0;

static long selection_start_time = 0, selection_duration = 0;
static long mouse_orig_time    = 0;
static SDL_Point mouse         = {-1, -1};
static int mouse_in_gantt_zone = 0;
static int mouse_down          = 0;

static int max_cores = -1, max_iterations = -1;
static long max_time = -1;

int use_thumbnails       = 1;
unsigned char brightness = 150;
unsigned soft_rendering  = 0;

static SDL_Color silver_color  = {192, 192, 192, 255};
static SDL_Color backgrd_color = {50, 50, 65, 255};

extern char *trace_dir[]; // Defined in main.c

struct
{
  SDL_Rect gantt, mosaic;
  SDL_Texture *label_tex;
  unsigned label_width, label_height;
  SDL_Texture **task_ids_tex;
  unsigned *task_ids_tex_width;
} trace_display_info[MAX_TRACES]; // no more than MAX_TRACES simultaneous traces

static SDL_Rect gantts_bounding_box;

struct
{
  int first_displayed_iter, last_displayed_iter;
} trace_ctrl[MAX_TRACES];

static inline unsigned layout_get_min_width (void)
{
  return WINDOW_MIN_WIDTH;
}

static unsigned layout_get_min_height (void)
{
  unsigned need_left, need_right, gantt_h;

  if (nb_traces == 1) {
    need_right = TOP_MARGIN + MIN_PREVIEW_DIM + BOTTOM_MARGIN;
    gantt_h    = trace[0].nb_cores * (MIN_TASK_HEIGHT + 2 * Y_MARGIN + 2);
    need_left  = TOP_MARGIN + gantt_h + BOTTOM_MARGIN;
  } else {
    need_right =
        TOP_MARGIN + 2 * MIN_PREVIEW_DIM + INTERTRACE_MARGIN + BOTTOM_MARGIN;
    gantt_h = (trace[0].nb_cores + trace[1].nb_cores) *
                  (MIN_TASK_HEIGHT + 2 * Y_MARGIN + 2) +
              INTERTRACE_MARGIN;
    need_left = TOP_MARGIN + gantt_h + BOTTOM_MARGIN;
  }
  return max (need_left, need_right);
}

static unsigned layout_get_max_height (void)
{
  unsigned need_left, need_right, gantt_h;

  if (nb_traces == 1) {
    need_right = TOP_MARGIN + MAX_PREVIEW_DIM + BOTTOM_MARGIN;
    gantt_h    = trace[0].nb_cores * (MAX_TASK_HEIGHT + 2 * Y_MARGIN + 2);
    need_left  = TOP_MARGIN + gantt_h + BOTTOM_MARGIN;
  } else {
    need_right =
        TOP_MARGIN + 2 * MAX_PREVIEW_DIM + INTERTRACE_MARGIN + BOTTOM_MARGIN;
    gantt_h = (trace[0].nb_cores + trace[1].nb_cores) *
                  (MAX_TASK_HEIGHT + 2 * Y_MARGIN + 2) +
              INTERTRACE_MARGIN;
    need_left = TOP_MARGIN + gantt_h + BOTTOM_MARGIN;
  }
  return max (need_left, need_right);
}

static void layout_place_buttons (void)
{
  quick_nav_rect.x = trace_display_info[0].gantt.x +
                     trace_display_info[0].gantt.w - quick_nav_rect.w;
  quick_nav_rect.y = 2;

  align_rect.x = quick_nav_rect.x - Y_MARGIN - align_rect.w;
  align_rect.y = 2;

  track_rect.x = align_rect.x - Y_MARGIN - track_rect.w;
  track_rect.y = 2;

  footprint_rect.x = track_rect.x - Y_MARGIN - footprint_rect.w;
  footprint_rect.y = 2;
}

static void layout_recompute (void)
{
  unsigned need_left;

  if (nb_traces == 1) {
    // Compute preview size
    PREVIEW_DIM = min (MAX_PREVIEW_DIM,
                       max (MIN_PREVIEW_DIM,
                            min (WINDOW_WIDTH / 4,
                                 WINDOW_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN)));

    // See how much space we have for GANTT chart
    unsigned space = WINDOW_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN;
    space /= trace[0].nb_cores;
    TASK_HEIGHT = space - 2 * Y_MARGIN - 2;
    if (TASK_HEIGHT < MIN_TASK_HEIGHT)
      exit_with_error ("Window height (%d) is not big enough to display so "
                       "many CPUS (%d)\n",
                       WINDOW_HEIGHT, trace[0].nb_cores);

    if (TASK_HEIGHT > MAX_TASK_HEIGHT)
      TASK_HEIGHT = MAX_TASK_HEIGHT;

    GANTT_HEIGHT = trace[0].nb_cores * CPU_ROW_HEIGHT;

    trace_display_info[0].gantt.x = LEFT_MARGIN;
    trace_display_info[0].gantt.y = TOP_MARGIN;
    trace_display_info[0].gantt.w = GANTT_WIDTH;
    trace_display_info[0].gantt.h = GANTT_HEIGHT;

    trace_display_info[0].mosaic.x = LEFT_MARGIN + GANTT_WIDTH + TOP_MARGIN / 2;
    trace_display_info[0].mosaic.y = TOP_MARGIN;
    trace_display_info[0].mosaic.w = PREVIEW_DIM;
    trace_display_info[0].mosaic.h = PREVIEW_DIM;
  } else {
    unsigned padding = 0;

    // Compute preview size
    PREVIEW_DIM =
        min (MAX_PREVIEW_DIM,
             max (MIN_PREVIEW_DIM,
                  min (WINDOW_WIDTH / 4, (WINDOW_HEIGHT - TOP_MARGIN -
                                          BOTTOM_MARGIN - INTERTRACE_MARGIN) /
                                             2)));

    // See how much space we have for GANTT chart
    unsigned space =
        WINDOW_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN - INTERTRACE_MARGIN;
    space /= (trace[0].nb_cores + trace[1].nb_cores);
    TASK_HEIGHT = space - 2 * Y_MARGIN - 2;
    if (TASK_HEIGHT < MIN_TASK_HEIGHT)
      exit_with_error ("Window height (%d) is not big enough to display so "
                       "many CPUS (%d)\n",
                       WINDOW_HEIGHT, trace[0].nb_cores);

    if (TASK_HEIGHT > MAX_TASK_HEIGHT)
      TASK_HEIGHT = MAX_TASK_HEIGHT;

    // First try with max task height
    GANTT_HEIGHT = (trace[0].nb_cores + trace[1].nb_cores) * CPU_ROW_HEIGHT +
                   INTERTRACE_MARGIN;
    need_left = TOP_MARGIN + GANTT_HEIGHT + BOTTOM_MARGIN;

    if (WINDOW_HEIGHT > need_left)
      padding = WINDOW_HEIGHT - need_left;

    trace_display_info[0].gantt.x = LEFT_MARGIN;
    trace_display_info[0].gantt.y = TOP_MARGIN + padding / 2;
    trace_display_info[0].gantt.w = GANTT_WIDTH;
    trace_display_info[0].gantt.h = trace[0].nb_cores * CPU_ROW_HEIGHT;

    trace_display_info[0].mosaic.x = LEFT_MARGIN + GANTT_WIDTH + TOP_MARGIN / 2;
    trace_display_info[0].mosaic.y = TOP_MARGIN;
    trace_display_info[0].mosaic.w = PREVIEW_DIM;
    trace_display_info[0].mosaic.h = PREVIEW_DIM;

    trace_display_info[1].gantt.x = LEFT_MARGIN;
    trace_display_info[1].gantt.y = TOP_MARGIN + INTERTRACE_MARGIN +
                                    trace_display_info[0].gantt.h + padding / 2;
    trace_display_info[1].gantt.w = GANTT_WIDTH;
    trace_display_info[1].gantt.h = trace[1].nb_cores * CPU_ROW_HEIGHT;

    trace_display_info[1].mosaic.w = PREVIEW_DIM;
    trace_display_info[1].mosaic.h = PREVIEW_DIM;
    trace_display_info[1].mosaic.x = LEFT_MARGIN + GANTT_WIDTH + TOP_MARGIN / 2;
    trace_display_info[1].mosaic.y =
        WINDOW_HEIGHT - BOTTOM_MARGIN - trace_display_info[1].mosaic.h;
  }

  gantts_bounding_box.x = trace_display_info[0].gantt.x;
  gantts_bounding_box.y = trace_display_info[0].gantt.y;
  gantts_bounding_box.w = trace_display_info[0].gantt.w;
  gantts_bounding_box.h = (trace_display_info[nb_traces - 1].gantt.y +
                           trace_display_info[nb_traces - 1].gantt.h - 1) -
                          gantts_bounding_box.y;

  // printf ("Window initial size: %dx%d\n", WINDOW_WIDTH, WINDOW_HEIGHT);
}

static inline int time_to_pixel (long time)
{
  return LEFT_MARGIN + time * GANTT_WIDTH / duration -
         start_time * GANTT_WIDTH / duration;
}

static inline long pixel_to_time (int x)
{
  return start_time + (x - LEFT_MARGIN) * duration / GANTT_WIDTH;
}

static inline int point_in_xrange (const SDL_Rect *r, int x)
{
  return x >= r->x && x < (r->x + r->w);
}

static inline int point_in_yrange (const SDL_Rect *r, int y)
{
  return y >= r->y && y < (r->y + r->h);
}

static inline int point_in_rect (const SDL_Point *p, const SDL_Rect *r)
{
  return point_in_xrange (r, p->x) && point_in_yrange (r, p->y);
}

static inline int point_inside_mosaic (const SDL_Point *p, unsigned trace_num)
{
  return point_in_rect (p, &trace_display_info[trace_num].mosaic);
}

static inline int point_inside_gantt (const SDL_Point *p, unsigned trace_num)
{
  return point_in_rect (p, &trace_display_info[trace_num].gantt);
}

static inline int point_inside_gantts (const SDL_Point *p)
{
  return point_in_rect (p, &gantts_bounding_box);
}

static inline int rects_do_intersect (const SDL_Rect *r1, const SDL_Rect *r2)
{
  return SDL_HasIntersection (r1, r2);
}

static inline void get_raw_rect (trace_task_t *t, SDL_Rect *dst)
{
  dst->x = t->x;
  dst->y = t->y;
  dst->w = t->w;
  dst->h = t->h;
}

static int get_y_mouse_sibbling (void)
{
  for (int t = 0; t < nb_traces; t++) {
    if (point_in_rect (&mouse, &trace_display_info[t].gantt)) {
      int dy = mouse.y - trace_display_info[t].gantt.y;
      if (dy < trace_display_info[1 - t].gantt.h)
        return trace_display_info[1 - t].gantt.y + dy;
      else
        return mouse.y;
    }
  }
  return mouse.y;
}

static inline int is_gpu (trace_t *const tr, int cpu_num)
{
  return cpu_num >= (tr->nb_cores - tr->nb_gpu);
}

static inline int is_lane (trace_t *const tr, int cpu_num)
{
  int tot = tr->nb_cores - tr->nb_gpu;
  if (cpu_num >= tot)
    return (cpu_num - tot) & 1; // #GPU is odd

  return 0;
}

// Texture creation functions

static void preload_thumbnails (unsigned nb_iter)
{
  if (use_thumbnails) {
    unsigned success = 0, expected = 0;
    unsigned nb_dirs  = 0;
    unsigned bound[2] = {nb_iter, nb_iter};
    char *dir[2]      = {trace_dir[0], trace_dir[1]};

    if (trace_dir[1] ||
        trace[0].first_iteration != trace[nb_traces - 1].first_iteration) {
      // Ok, we have to use two separate arrays to store textures,
      // either because thumbnails are located in two separate folders
      // or because we compare two traces starting from a different iteration
      // number. (Note that in this latter case -- and if iteration ranges
      // overlap -- we could probably try to load once and shift the indexes in
      // the second array... I don't think it is relevant)

      nb_dirs = 2;
      if (!dir[1])
        dir[1] = dir[0];
      bound[0]     = trace[0].nb_iterations;
      bound[1]     = trace[1].nb_iterations;
      expected     = bound[0] + bound[1];
      thumb_tex[0] = malloc (bound[0] * sizeof (SDL_Texture *));
      thumb_tex[1] = malloc (bound[1] * sizeof (SDL_Texture *));
    } else {
      // We use a unique array to store thumbnails, either because we're
      // displaying a single trace or because we compare two traces with one
      // iteration range being a subset of the other (in this case, nb_iter is
      // the maximum number of iterations).
      nb_dirs      = 1;
      expected     = bound[0];
      thumb_tex[0] = malloc (nb_iter * sizeof (SDL_Texture *));
      thumb_tex[1] = thumb_tex[0];
    }

    for (int d = 0; d < nb_dirs; d++)
      for (int iter = 0; iter < bound[d]; iter++) {
        SDL_Surface *thumb = NULL;
        char filename[1024];

        sprintf (filename, "%s/thumb_%04d.png", dir[d],
                 trace[d].first_iteration + iter);
        thumb = IMG_Load (filename);

        if (thumb != NULL) {
          success++;

          thumb_tex[d][iter] = SDL_CreateTextureFromSurface (renderer, thumb);
          SDL_FreeSurface (thumb);
          SDL_SetTextureAlphaMod (thumb_tex[d][iter], brightness);
        } else
          thumb_tex[d][iter] = NULL;
      }

    printf ("%d/%u thumbnails successfully preloaded\n", success, expected);
  }
}

static void create_task_textures (unsigned nb_cores)
{
  Uint32 *restrict img = malloc (GANTT_WIDTH * TASK_HEIGHT * sizeof (Uint32));

  perf_fill = malloc ((MAX_COLORS + 1) * sizeof (SDL_Texture *));

  SDL_Surface *s = SDL_CreateRGBSurfaceFrom (
      img, GANTT_WIDTH, TASK_HEIGHT, 32, GANTT_WIDTH * sizeof (Uint32),
      0xff000000, 0x00ff0000, 0x0000ff00, 0x000000ff);
  if (s == NULL)
    exit_with_error ("SDL_CreateRGBSurfaceFrom () failed");

  unsigned largeur_couleur_origine = GANTT_WIDTH / 4;
  unsigned largeur_degrade         = GANTT_WIDTH - largeur_couleur_origine;
  float attenuation_depart         = 1.0;
  float attenuation_finale         = 0.3;

  for (int c = 0; c < MAX_COLORS + 1; c++) {
    bzero (img, GANTT_WIDTH * TASK_HEIGHT * sizeof (Uint32));

    if (c == MAX_COLORS) // special treatment for white color
      attenuation_finale = 0.5;

    for (int j = 0; j < GANTT_WIDTH; j++) {
      unsigned couleur = cpu_colors[c];
      unsigned r       = couleur >> 24;
      unsigned g       = couleur >> 16 & 255;
      unsigned b       = couleur >> 8 & 255;

      if (j >= largeur_couleur_origine) {
        float coef =
            attenuation_depart -
            ((((float)(j - largeur_couleur_origine)) / largeur_degrade)) *
                (attenuation_depart - attenuation_finale);
        r = r * coef;
        g = g * coef;
        b = b * coef;
      }

      for (int i = 0; i < TASK_HEIGHT; i++)
        img[i * GANTT_WIDTH + j] = (r << 24) | (g << 16) | (b << 8) | 255;
    }

    perf_fill[c] = SDL_CreateTextureFromSurface (renderer, s);
  }

  SDL_FreeSurface (s);
  free (img);

  s = SDL_CreateRGBSurface (0, SQUARE_SIZE, SQUARE_SIZE, 32, 0xff000000,
                            0x00ff0000, 0x0000ff00, 0x000000ff);
  if (s == NULL)
    exit_with_error ("SDL_CreateRGBSurface () failed");

  SDL_FillRect (s, NULL, 0x000000FF); // back
  black_square = SDL_CreateTextureFromSurface (renderer, s);

  SDL_FillRect (s, NULL, 0xFFFFFFFF); // white
  white_square = SDL_CreateTextureFromSurface (renderer, s);

  square_tex_bright = malloc ((MAX_COLORS + 1) * sizeof (SDL_Texture *));
  square_tex_dark   = malloc ((MAX_COLORS + 1) * sizeof (SDL_Texture *));

  for (int c = 0; c < MAX_COLORS + 1; c++) {
    SDL_FillRect (s, NULL, cpu_colors[c]);
    square_tex_bright[c] = SDL_CreateTextureFromSurface (renderer, s);
    square_tex_dark[c]   = SDL_CreateTextureFromSurface (renderer, s);
    // Dark tiles are semi-transparent
    SDL_SetTextureAlphaMod (square_tex_dark[c], TILE_ALPHA);
  }

  SDL_FreeSurface (s);
}

static void create_digit_textures (TTF_Font *font)
{
  SDL_Color white_color = {255, 255, 255, 255};
  SDL_Surface *s        = NULL;

  for (int c = 0; c < 10; c++) {
    char msg[32];
    snprintf (msg, 32, "%d", c);

    s = TTF_RenderText_Blended (font, msg, white_color);
    if (s == NULL)
      exit_with_error ("TTF_RenderText_Solid failed: %s", SDL_GetError ());

    digit_tex_width[c] = s->w;
    digit_tex_height   = s->h;
    digit_tex[c]       = SDL_CreateTextureFromSurface (renderer, s);

    SDL_FreeSurface (s);
  }

  s = TTF_RenderUTF8_Blended (font, "µs", white_color);
  if (s == NULL)
    exit_with_error ("TTF_RenderText_Solid failed: %s", SDL_GetError ());

  us_tex = SDL_CreateTextureFromSurface (renderer, s);
  SDL_FreeSurface (s);

  s = TTF_RenderUTF8_Blended (font, "Σ: ", white_color);
  if (s == NULL)
    exit_with_error ("TTF_RenderText_Solid failed: %s", SDL_GetError ());

  sigma_tex = SDL_CreateTextureFromSurface (renderer, s);
  SDL_FreeSurface (s);
}

static void create_tab_textures (TTF_Font *font)
{
  SDL_Surface *surf = IMG_Load ("./traces/img/tab-left.png");
  if (surf == NULL)
    exit_with_error ("IMG_Load failed: %s", SDL_GetError ());

  tab_left = SDL_CreateTextureFromSurface (renderer, surf);
  SDL_FreeSurface (surf);

  surf = IMG_Load ("./traces/img/tab-high.png");
  if (surf == NULL)
    exit_with_error ("IMG_Load failed: %s", SDL_GetError ());

  tab_high = SDL_CreateTextureFromSurface (renderer, surf);
  SDL_FreeSurface (surf);

  surf = IMG_Load ("./traces/img/tab-right.png");
  if (surf == NULL)
    exit_with_error ("IMG_Load failed: %s", SDL_GetError ());

  tab_right = SDL_CreateTextureFromSurface (renderer, surf);
  SDL_FreeSurface (surf);

  surf = IMG_Load ("./traces/img/tab-low.png");
  if (surf == NULL)
    exit_with_error ("IMG_Load failed: %s", SDL_GetError ());

  tab_low = SDL_CreateTextureFromSurface (renderer, surf);
  SDL_FreeSurface (surf);

  for (int t = 0; t < nb_traces; t++) {
    SDL_Surface *s =
        TTF_RenderUTF8_Blended (font, trace[t].label, backgrd_color);
    if (s == NULL)
      exit_with_error ("TTF_RenderUTF8_Blended failed: %s", SDL_GetError ());

    trace_display_info[t].label_tex =
        SDL_CreateTextureFromSurface (renderer, s);
    trace_display_info[t].label_width  = s->w;
    trace_display_info[t].label_height = s->h;
    SDL_FreeSurface (s);
  }
}

static void create_task_ids_textures (TTF_Font *font)
{
  for (int t = 0; t < nb_traces; t++) {
    trace_display_info[t].task_ids_tex =
        calloc (trace[t].task_ids_count, sizeof (SDL_Texture *));
    trace_display_info[t].task_ids_tex_width =
        calloc (trace[t].task_ids_count, sizeof (unsigned));

    for (int i = 0; i < trace[t].task_ids_count; i++) {
      SDL_Surface *s =
          TTF_RenderUTF8_Blended (font, trace[t].task_ids[i], silver_color);
      if (s == NULL)
        exit_with_error ("TTF_RenderUTF8_Blended failed: %s", SDL_GetError ());

      trace_display_info[t].task_ids_tex[i] =
          SDL_CreateTextureFromSurface (renderer, s);
      trace_display_info[t].task_ids_tex_width[i] = s->w;
      SDL_FreeSurface (s);
    }
  }
}

static void blit_on_surface (SDL_Surface *surface, TTF_Font *font, int trace,
                             unsigned line, char *msg)
{
  SDL_Rect dst;
  SDL_Surface *s = TTF_RenderText_Blended (font, msg, silver_color);
  if (s == NULL)
    exit_with_error ("TTF_RenderText_Blended failed: %s", SDL_GetError ());

  dst.x = LEFT_MARGIN - s->w;
  dst.y = trace_display_info[trace].gantt.y + CPU_ROW_HEIGHT * line +
          CPU_ROW_HEIGHT / 2 - (s->h / 2);

  SDL_BlitSurface (s, NULL, surface, &dst);
  SDL_FreeSurface (s);
}

static void create_cpu_textures (TTF_Font *font)
{
  SDL_Surface *surface =
      SDL_CreateRGBSurface (0, LEFT_MARGIN, WINDOW_HEIGHT, 32, 0xff000000,
                            0x00ff0000, 0x0000ff00, 0x000000ff);
  if (surface == NULL)
    exit_with_error ("SDL_CreateRGBSurface failed: %s", SDL_GetError ());

  for (int t = 0; t < nb_traces; t++) {
    for (int c = 0; c < trace[t].nb_cores; c++) {
      char msg[32];
      if (is_gpu (trace + t, c)) {
        int lane = c - (trace[t].nb_cores - trace[t].nb_gpu);
        int gpu  = lane >> 1;
        if (lane & 1)
          snprintf (msg, 32, "<-> G %d ", gpu);
        else
          snprintf (msg, 32, "GPU %2d ", gpu);
      } else
        snprintf (msg, 32, "CPU %2d ", c);

      blit_on_surface (surface, font, t, c, msg);
    }
  }

  if (text_texture == NULL)
    SDL_DestroyTexture (text_texture);

  text_texture = SDL_CreateTextureFromSurface (renderer, surface);
  if (text_texture == NULL)
    exit_with_error ("SDL_CreateTexture failed: %s", SDL_GetError ());

  SDL_FreeSurface (surface);
}

static void create_text_texture (TTF_Font *font)
{
  create_cpu_textures (font);

  create_digit_textures (font);

  create_tab_textures (font);

  create_task_ids_textures (font);
}

static void create_misc_tex (void)
{
  SDL_Surface *surf = SDL_CreateRGBSurface (0, 2, WINDOW_HEIGHT, 32, 0xff000000,
                                            0x00ff0000, 0x0000ff00, 0x000000ff);
  if (surf == NULL)
    exit_with_error ("SDL_CreateRGBSurface failed: %s", SDL_GetError ());

  SDL_FillRect (surf, NULL, SDL_MapRGB (surf->format, 0, 255, 255));
  vertical_line = SDL_CreateTextureFromSurface (renderer, surf);
  SDL_SetTextureBlendMode (vertical_line, SDL_BLENDMODE_BLEND);
  SDL_FreeSurface (surf);

  surf = SDL_CreateRGBSurface (0, GANTT_WIDTH, 2, 32, 0xff000000, 0x00ff0000,
                               0x0000ff00, 0x000000ff);
  if (surf == NULL)
    exit_with_error ("SDL_CreateRGBSurface failed: %s", SDL_GetError ());

  SDL_FillRect (surf, NULL, SDL_MapRGB (surf->format, 0, 255, 255));
  horizontal_line = SDL_CreateTextureFromSurface (renderer, surf);
  SDL_SetTextureBlendMode (horizontal_line, SDL_BLENDMODE_BLEND);

  SDL_FillRect (surf, NULL, SDL_MapRGB (surf->format, 150, 150, 200));
  horizontal_bis = SDL_CreateTextureFromSurface (renderer, surf);
  SDL_SetTextureBlendMode (horizontal_bis, SDL_BLENDMODE_BLEND);

  SDL_FreeSurface (surf);

  surf = IMG_Load ("./traces/img/frame.png");
  if (surf == NULL)
    exit_with_error ("IMG_Load failed: %s", SDL_GetError ());

  BUBBLE_WIDTH  = surf->w;
  BUBBLE_HEIGHT = surf->h;

  bulle_tex = SDL_CreateTextureFromSurface (renderer, surf);
  SDL_FreeSurface (surf);

  surf = IMG_Load ("./traces/img/quick-nav.png");
  if (surf == NULL)
    exit_with_error ("IMG_Load failed: %s", SDL_GetError ());

  quick_nav_tex = SDL_CreateTextureFromSurface (renderer, surf);
  SDL_FreeSurface (surf);
  SDL_SetTextureAlphaMod (quick_nav_tex, quick_nav_mode ? 0xFF : BUTTON_ALPHA);

  SDL_QueryTexture (quick_nav_tex, NULL, NULL, &quick_nav_rect.w,
                    &quick_nav_rect.h);

  surf = IMG_Load ("./traces/img/auto-align.png");
  if (surf == NULL)
    exit_with_error ("IMG_Load failed: %s", SDL_GetError ());

  align_tex = SDL_CreateTextureFromSurface (renderer, surf);
  SDL_FreeSurface (surf);
  SDL_SetTextureAlphaMod (align_tex,
                          trace_data_align_mode ? 0xFF : BUTTON_ALPHA);

  SDL_QueryTexture (align_tex, NULL, NULL, &align_rect.w, &align_rect.h);

  surf = IMG_Load ("./traces/img/track-mode.png");
  if (surf == NULL)
    exit_with_error ("IMG_Load failed: %s", SDL_GetError ());

  track_tex = SDL_CreateTextureFromSurface (renderer, surf);
  SDL_FreeSurface (surf);
  SDL_SetTextureAlphaMod (track_tex,
                          tracking_mode ? 0xFF : BUTTON_ALPHA);

  SDL_QueryTexture (track_tex, NULL, NULL, &track_rect.w, &track_rect.h);

  surf = IMG_Load ("./traces/img/footprint.png");
  if (surf == NULL)
    exit_with_error ("IMG_Load failed: %s", SDL_GetError ());

  footprint_tex = SDL_CreateTextureFromSurface (renderer, surf);
  SDL_FreeSurface (surf);
  SDL_SetTextureAlphaMod (footprint_tex,
                          footprint_mode ? 0xFF : BUTTON_ALPHA);

  SDL_QueryTexture (footprint_tex, NULL, NULL, &footprint_rect.w, &footprint_rect.h);
}

// Display functions

static inline int get_tile_rect (trace_t *tr, trace_task_t *t, SDL_Rect *dst)
{
  if (t->w && t->h) {
    dst->x = t->x * trace_display_info[tr->num].mosaic.w / tr->dimensions +
             trace_display_info[tr->num].mosaic.x;
    dst->y = t->y * trace_display_info[tr->num].mosaic.h / tr->dimensions +
             trace_display_info[tr->num].mosaic.y;
    dst->w =
        ((t->x + t->w) * trace_display_info[tr->num].mosaic.w / tr->dimensions +
         trace_display_info[tr->num].mosaic.x - dst->x)
            ?: 1;
    dst->h =
        ((t->y + t->h) * trace_display_info[tr->num].mosaic.h / tr->dimensions +
         trace_display_info[tr->num].mosaic.y - dst->y)
            ?: 1;
    return 1;
  } else { // task has no associated tile
    dst->x = 0;
    dst->y = 0;
    dst->w = 0;
    dst->h = 0;
    return 0;
  }
}

static void show_tile (trace_t *tr, trace_task_t *t, unsigned cpu,
                       unsigned highlight)
{
  SDL_Rect dst;
  unsigned task_color =
      is_lane (tr, cpu) ? gpu_index[t->task_type] : (cpu % MAX_COLORS);

  get_tile_rect (tr, t, &dst);

  SDL_RenderCopy (
      renderer,
      (highlight ? square_tex_bright[task_color] : square_tex_dark[task_color]),
      NULL, &dst);
}

static void display_tab (unsigned trace_num)
{
  SDL_Rect dst;

  dst.x = trace_display_info[trace_num].gantt.x;
  dst.y = trace_display_info[trace_num].gantt.y - 24;
  dst.w = 32;
  dst.h = 24;
  SDL_RenderCopy (renderer, tab_left, NULL, &dst);

  dst.x += 8;
  dst.y = trace_display_info[trace_num].gantt.y - 24;
  dst.w = trace_display_info[trace_num].label_width;
  dst.h = 24;
  SDL_RenderCopy (renderer, tab_high, NULL, &dst);

  dst.y += 2;
  dst.w = trace_display_info[trace_num].label_width;
  dst.h = trace_display_info[trace_num].label_height;
  SDL_RenderCopy (renderer, trace_display_info[trace_num].label_tex, NULL,
                  &dst);

  dst.x += dst.w;
  dst.y -= 2;
  dst.w = 32;
  dst.h = 24;
  SDL_RenderCopy (renderer, tab_right, NULL, &dst);

  dst.x += 32;
  dst.w =
      trace_display_info[trace_num].gantt.w; // We don't care, thanks clipping!
  SDL_RenderCopy (renderer, tab_low, NULL, &dst);
}

static void display_text (void)
{
  SDL_Rect dst;

  dst.x = 0;
  dst.y = 0;
  dst.w = LEFT_MARGIN;
  dst.h = WINDOW_HEIGHT;

  SDL_RenderCopy (renderer, text_texture, NULL, &dst);
}

static void display_iter_number (unsigned iter, unsigned y_offset,
                                 unsigned x_offset, unsigned max_size)
{
  unsigned digits[10];
  unsigned nbd = 0, width;
  SDL_Rect dst;

  do {
    digits[nbd] = iter % 10;
    iter /= 10;
    nbd++;
  } while (iter > 0);

  width = nbd * digit_tex_width[0]; // approx

  dst.x = x_offset + max_size / 2 - width / 2;
  dst.y = y_offset;
  dst.h = digit_tex_height;

  for (int d = nbd - 1; d >= 0; d--) {
    unsigned the_digit = digits[d];
    dst.w              = digit_tex_width[the_digit];

    SDL_RenderCopy (renderer, digit_tex[the_digit], NULL, &dst);

    dst.x += digit_tex_width[the_digit];
  }
}

static void display_duration (unsigned long task_duration, unsigned x_offset,
                              unsigned y_offset, unsigned max_size,
                              unsigned with_sigma)
{
  unsigned digits[10];
  unsigned nbd = 0, width;
  SDL_Rect dst;

  do {
    digits[nbd] = task_duration % 10;
    task_duration /= 10;
    nbd++;
  } while (task_duration > 0);

  width = (nbd + 2 + 2 * with_sigma) * digit_tex_width[0]; // approx

  dst.x = x_offset + max_size / 2 - width / 2;
  dst.y = y_offset;
  dst.h = digit_tex_height;

  if (with_sigma) {
    dst.w = 19;
    SDL_RenderCopy (renderer, sigma_tex, NULL, &dst);
    dst.x += dst.w;
  }

  dst.w = digit_tex_width[0];

  for (int d = nbd - 1; d >= 0; d--) {
    unsigned the_digit = digits[d];

    SDL_RenderCopy (renderer, digit_tex[the_digit], NULL, &dst);

    dst.x += digit_tex_width[the_digit];
  }

  dst.w = 18;
  SDL_RenderCopy (renderer, us_tex, NULL, &dst);
}

static void display_selection (void)
{
  if (selection_duration > 0) {
    SDL_Rect dst;

    dst.x = time_to_pixel (selection_start_time);
    dst.y = trace_display_info[0].gantt.y;
    dst.w =
        time_to_pixel (selection_start_time + selection_duration) - dst.x + 1;
    dst.h = GANTT_HEIGHT;

    SDL_SetRenderDrawColor (renderer, 255, 255, 255, 100);
    SDL_RenderFillRect (renderer, &dst);

    display_duration (selection_duration, dst.x + dst.w / 2, dst.y + dst.h / 2,
                      0, 0);
  }
}

static void display_bubble (int x, int y, unsigned long duration,
                            unsigned with_sigma)
{
  SDL_Rect dst;

  dst.x = x - (BUBBLE_WIDTH >> 1);
  dst.y = y - BUBBLE_HEIGHT;
  dst.w = BUBBLE_WIDTH;
  dst.h = BUBBLE_HEIGHT;

  SDL_RenderCopy (renderer, bulle_tex, NULL, &dst);

  display_duration (duration, x - (BUBBLE_WIDTH >> 1) + 1, dst.y + 1, dst.w - 3,
                    with_sigma);
}

typedef struct
{
  trace_task_t *task;
  trace_t *trace;
  unsigned iter;
  long cumulated_duration;
  SDL_Rect area;
} selected_task_info_t;

#define SELECTED_TASK_INFO_INITIALIZER                                         \
  {                                                                            \
    NULL, NULL, 0, 0,                                                          \
    {                                                                          \
      0, 0, 0, 0                                                               \
    }                                                                          \
  }

static void selected_task_info_init (selected_task_info_t *info)
{
  info->task               = NULL;
  info->trace              = NULL;
  info->cumulated_duration = 0;
}

static void display_mouse_selection (const selected_task_info_t *selected)
{
  SDL_Rect dst;

  if (horiz_mode) {
    if (!footprint_mode) {
      // horizontal bar
      if (mouse_in_gantt_zone) {
        dst.x = trace_display_info[0].gantt.x;
        dst.y = mouse.y;
        dst.w = GANTT_WIDTH;
        dst.h = 1;

        SDL_RenderCopy (renderer, horizontal_line, NULL, &dst);

        dst.y = get_y_mouse_sibbling ();
        if (dst.y != mouse.y)
          SDL_RenderCopy (renderer, horizontal_bis, NULL, &dst);
      }
    }
  } else {
    trace_t *tr     = selected->trace;
    trace_task_t *t = selected->task;

    // vertical bar
    if (mouse_in_gantt_zone) {
      dst.x = mouse.x;
      dst.w = 1;
      if (tracking_mode) {
        if (tr != NULL) {
          dst.x = mouse.x;
          dst.y = trace_display_info[tr->num].gantt.y;
          dst.w = 1;
          dst.h = trace_display_info[tr->num].gantt.h;
          SDL_RenderCopy (renderer, vertical_line, NULL, &dst);
        }
      } else {
        dst.x = mouse.x;
        dst.y = trace_display_info[0].gantt.y;
        dst.w = 1;
        dst.h = GANTT_HEIGHT;
        SDL_RenderCopy (renderer, vertical_line, NULL, &dst);
      }

      if (t != NULL) {
        display_bubble (mouse.x, trace_display_info[tr->num].gantt.y - 4,
                        t->end_time - t->start_time, 0);

        if (tracking_mode) {
          // int iter = selected->iter - tr->first_iteration;
          // int x = (time_to_pixel (iteration_start_time (tr, iter)) +
          //         time_to_pixel (iteration_end_time (tr, iter))) >> 1;
          int y = trace_display_info[1 - tr->num].gantt.y - 4;
          display_bubble (mouse.x, y, selected->cumulated_duration, 1);
        }

        if (t->task_id) { // Do not display "anonymous" IDs
          dst.w = trace_display_info[tr->num].task_ids_tex_width[t->task_id];
          dst.h = FONT_HEIGHT;
          dst.x = mouse.x - dst.w / 2;
          dst.y = trace_display_info[tr->num].gantt.y +
                  trace_display_info[tr->num].gantt.h;
          SDL_RenderCopy (renderer,
                          trace_display_info[tr->num].task_ids_tex[t->task_id],
                          NULL, &dst);
        }
      }
    }
  }
}

static void display_misc_status (void)
{
  SDL_RenderCopy (renderer, quick_nav_tex, NULL, &quick_nav_rect);
  SDL_RenderCopy (renderer, footprint_tex, NULL, &footprint_rect);

  if (nb_traces > 1) {
    SDL_RenderCopy (renderer, align_tex, NULL, &align_rect);
    SDL_RenderCopy (renderer, track_tex, NULL, &track_rect);
  }
}

static void display_tile_background (int tr)
{
#ifdef PRELOAD_THUMBNAILS
  static int displayed_iter[MAX_TRACES] = {-1, -1};
  static SDL_Texture *tex[MAX_TRACES]   = {NULL};

  SDL_RenderCopy (renderer, black_square, NULL, &trace_display_info[tr].mosaic);

  if (use_thumbnails) {

    if (mouse_in_gantt_zone) {
      long time = pixel_to_time (mouse.x);
      int iter  = trace_data_search_iteration (&trace[tr], time);

      if (iter != -1 && iter != displayed_iter[tr]) {
        displayed_iter[tr] = iter;
        tex[tr]            = thumb_tex[tr][iter];
      }
    }
  }

  if (tex[tr] != NULL)
    SDL_RenderCopy (renderer, tex[tr], NULL, &trace_display_info[tr].mosaic);

#else
#error Obsolete code needs to be fixed! (or dropped)
  if (use_thumbnails) {
    static int displayed_iter = -1;
    static SDL_Surface *thumb = NULL;
    char filename[1024];

    if (mouse.x != -1) {
      long time = pixel_to_time (mouse.x);
      int iter  = trace_data_search_iteration (tr, time);

      if (iter != -1 && iter != displayed_iter) {
        displayed_iter = iter;
        if (thumb != NULL) {
          SDL_FreeSurface (thumb);
          thumb = NULL;
        }
        sprintf (filename, "./traces/data/thumb_%04d.png", iter + 1);
        thumb = IMG_Load (filename);
        // if( thumb != NULL)
        //   printf ("Thumbnail [%s] loaded\n", filename);
      }
    }

    if (thumb != NULL) {
      // TODO: display thumb
    } else {
      // TODO; display black square
    }
  }
#endif
}

static void display_gantt_background (trace_t *tr, int _t, int first_it)
{
  display_tab (_t);

  // Display iterations' background and number
  for (unsigned it = first_it;
       (it < tr->nb_iterations) && (iteration_start_time (tr, it) < end_time);
       it++) {
    SDL_Rect r;

    r.x = time_to_pixel (iteration_start_time (tr, it));
    r.y = trace_display_info[_t].gantt.y;
    r.w = time_to_pixel (iteration_end_time (tr, it)) - r.x + 1;
    r.h = trace_display_info[_t].gantt.h;

    // Background of iterations is black
    SDL_SetRenderDrawColor (renderer, 0, 0, 0, 255);
    SDL_RenderFillRect (renderer, &r);

    if (trace_data_align_mode && tr->iteration[it].gap > 0) {
      SDL_Rect gap;

      gap.x =
          time_to_pixel (iteration_end_time (tr, it) - tr->iteration[it].gap);
      gap.y = r.y;
      gap.w = r.x + r.w - gap.x;
      gap.h = r.h;

      SDL_SetRenderDrawColor (renderer, 0, 90, 0, 255);
      SDL_RenderFillRect (renderer, &gap);
    }

    display_iter_number (tr->first_iteration + it,
                         trace_display_info[_t].gantt.y +
                             trace_display_info[_t].gantt.h + 1,
                         r.x, r.w);
  }

  // Draw the separation lines
  {
    SDL_Rect dst = {LEFT_MARGIN,
                    trace_display_info[_t].gantt.y + CPU_ROW_HEIGHT - 2,
                    GANTT_WIDTH, 2};

    for (int c = 0; c < tr->nb_cores; c++) {
      SDL_RenderCopy (renderer, perf_fill[c % MAX_COLORS], NULL, &dst);
      dst.y += CPU_ROW_HEIGHT;
    }
  }
}

static void trace_graphics_display_trace (unsigned _t,
                                          selected_task_info_t *selected)
{
  trace_t *const tr       = trace + _t;
  const unsigned first_it = trace_ctrl[_t].first_displayed_iter - 1;
  trace_task_t *to_be_emphasized[max_cores];
  SDL_Rect target_tile_rect; // Only used when mouse is over Mosaic
  int target_tile = 0;       // Only used when mouse is over Mosaic
  int cpu         = 0;       // Only used in tracking mode
  unsigned wh     = trace_display_info[_t].gantt.y + Y_MARGIN;

  bzero (to_be_emphasized, max_cores * sizeof (trace_task_t *));

  // Set clipping region
  {
    SDL_Rect clip = trace_display_info[0].gantt;

    // We enlarge the clipping area along the y-axis to enable display of
    // iteration numbers
    clip.y = 0;
    clip.h = WINDOW_HEIGHT;

    SDL_RenderSetClipRect (renderer, &clip);
  }

  display_gantt_background (tr, _t, first_it);

  SDL_Point virt_mouse = mouse;
  int in_mosaic        = 0;

  // Normalize (virt_mouse.x, virt_mouse.y) mouse coordinates
  if (mouse_in_gantt_zone) {
    if (point_inside_gantt (&mouse, _t))
      selected->trace = tr;

    if (horiz_mode && point_inside_gantt (&mouse, 1 - _t)) {
      virt_mouse.y = get_y_mouse_sibbling ();
      virt_mouse.x = -1;
    }
  } else {
    if (point_inside_mosaic (&mouse, _t)) {
      // Mouse is over our tile mosaic
      in_mosaic = 1;
    } else if ((nb_traces > 1) && point_inside_mosaic (&mouse, 1 - _t)) {
      // Mouse is over the other tile mosaic
      in_mosaic    = 1;
      virt_mouse.x = trace_display_info[_t].mosaic.x +
                     (mouse.x - trace_display_info[1 - _t].mosaic.x);
      virt_mouse.y = trace_display_info[_t].mosaic.y +
                     (mouse.y - trace_display_info[1 - _t].mosaic.y);
    }
  }

  // We go through the range of iterations and we display tasks & associated
  // tiles
  if (first_it < tr->nb_iterations)
    for (int c = 0; c < tr->nb_cores; c++) {
      // We get a pointer on the first task executed by
      // CPU 'c' at first displayed iteration
      trace_task_t *first = tr->iteration[first_it].first_cpu_task[c];
      unsigned task_color = c % MAX_COLORS;

      if (first != NULL)
        // We follow the list of tasks, starting from this first task
        list_for_each_entry_from (trace_task_t, t, tr->per_cpu + c, first,
                                  cpu_chain)
        {
          if (task_end_time (tr, t) < start_time)
            continue;

          // We stop if we encounter a task belonging to a greater iteration
          if (task_start_time (tr, t) > end_time)
            break;

          // Ok, this task should appear on the screen
          SDL_Rect dst;

          // Project the task in the Gantt chart
          dst.x = time_to_pixel (task_start_time (tr, t));
          dst.y = wh;
          dst.w = time_to_pixel (task_end_time (tr, t)) - dst.x + 1;
          dst.h = TASK_HEIGHT;

          // If task is a GPU tranfer lane, modify height, y-offset and color
          if (is_lane (tr, c)) {
            dst.h = TASK_HEIGHT / 2;
            if (t->task_type == TASK_TYPE_READ)
              dst.y += TASK_HEIGHT / 2;

            task_color = gpu_index[t->task_type];
          }

          // Check if mouse is within the bounds of the gantt zone
          if (mouse_in_gantt_zone) {

            if (horiz_mode && (footprint_mode | point_in_yrange (&dst, virt_mouse.y))) {
              if (to_be_emphasized[c] == NULL)
                to_be_emphasized[c] =
                    first; // store a ref to the first task on this lane
            }

            if (point_in_xrange (&dst, virt_mouse.x)) {
              // vertical line crosses the task
              if (point_in_yrange (&dst, virt_mouse.y)) {

                selected->task = t;

                if (tracking_mode) {
                  cpu            = c;
                  selected->iter = tr->first_iteration + t->iteration;
                  get_raw_rect (t, &selected->area);
                }
                // The task is under the mouse cursor: display it a little
                // bigger!
                dst.x -= 3;
                dst.y -= 3;
                dst.w += 6;
                dst.h += 6;
              }

              if (!horiz_mode && !tracking_mode && t->w &&
                  t->h) // task has as associated tile to be displayed
                to_be_emphasized[c] = t;
            }

            // If tracking mode is enabled, we highlight tasks which work on
            // tiles intersecting the working set of selected task
            if (tracking_mode && selected->task != NULL &&
                _t != selected->trace->num &&
                selected->iter == t->iteration + tr->first_iteration &&
                selected->task->task_id == t->task_id) {
              SDL_Rect r;

              get_raw_rect (t, &r);
              if (rects_do_intersect (&r, &selected->area)) {
                selected->cumulated_duration += t->end_time - t->start_time;

                SDL_RenderCopy (renderer, perf_fill[MAX_COLORS], NULL,
                                &dst); // white
                if (to_be_emphasized[c] == NULL)
                  to_be_emphasized[c] = t;
              } else
                SDL_RenderCopy (renderer, perf_fill[task_color], NULL, &dst);
            } else
              SDL_RenderCopy (renderer, perf_fill[task_color], NULL, &dst);

          } else if (in_mosaic) {
            SDL_Rect r;

            get_tile_rect (tr, t, &r);

            if (point_in_rect (&virt_mouse, &r)) {
              if (!target_tile) {
                target_tile      = 1;
                target_tile_rect = r;
              }
              SDL_RenderCopy (renderer, perf_fill[MAX_COLORS], NULL,
                              &dst); // white
            } else
              SDL_RenderCopy (renderer, perf_fill[task_color], NULL, &dst);
          } else
            SDL_RenderCopy (renderer, perf_fill[task_color], NULL, &dst);
        }

      wh += CPU_ROW_HEIGHT;
    }

  // Display mouse selection rectangle (if any)
  display_selection ();

  // Disable clipping region
  SDL_RenderSetClipRect (renderer, NULL);

  display_tile_background (_t);

  if (target_tile)
    // Display mouse-selected tile in white color
    SDL_RenderCopy (renderer, square_tex_dark[MAX_COLORS], NULL,
                    &target_tile_rect);
  else if (horiz_mode) {
    for (int c = 0; c < tr->nb_cores; c++) {

      if (to_be_emphasized[c] != NULL) {
        // We follow the list of tasks, starting from this first task
        list_for_each_entry_from (trace_task_t, t, tr->per_cpu + c,
                                  to_be_emphasized[c], cpu_chain)
        {
          // Skip if the task has no associated tile
          if (t->w == 0 || t->h == 0)
            continue;

          if (task_end_time (tr, t) < start_time)
            continue;

          // We stop if we encounter a task belonging to a greater iteration
          if (task_start_time (tr, t) > end_time)
            break;

          // Ok, this task should have its tile displayed
          show_tile (tr, t, c, (t == selected->task));
        }
      }
    }
  } else if (tracking_mode) {
    if (selected->task != NULL && selected->task->w && selected->task->h) {
      if (selected->trace->num == _t) {
        show_tile (tr, selected->task, cpu, 1);
      } else {
        int raw_iter = selected->iter - tr->first_iteration;

        for (int c = 0; c < tr->nb_cores; c++) {
          if (to_be_emphasized[c] != NULL)
            // We follow the list of tasks, starting from this first task
            list_for_each_entry_from (trace_task_t, t, tr->per_cpu + c,
                                      to_be_emphasized[c], cpu_chain)
            {
              if (task_end_time (tr, t) < start_time)
                continue;

              // Skip if the task has no associated tile
              if (t->w == 0 || t->h == 0)
                continue;

              // Stop when reaching next iteration
              if (t->iteration > raw_iter)
                break;

              // Stop if we encounter a task not displayed on screen
              if (task_start_time (tr, t) > end_time)
                break;

              SDL_Rect r;

              get_raw_rect (t, &r);
              if (rects_do_intersect (&r, &selected->area))
                show_tile (tr, t, c, 1);
            }
        }
      }
    }
  } else {
    // Display tiles corresponding to tasks intersecting with mouse "iso x"
    // axis
    for (int c = 0; c < tr->nb_cores; c++) {
      if (to_be_emphasized[c] != NULL) {
        show_tile (tr, to_be_emphasized[c], c,
                   (to_be_emphasized[c] == selected->task));
      }
    }
  }
}

static void trace_graphics_display (void)
{
  selected_task_info_t selected = SELECTED_TASK_INFO_INITIALIZER;

  SDL_RenderClear (renderer);

  // Draw the dark grey background
  {
    SDL_Rect all = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};

    SDL_SetRenderDrawColor (renderer, backgrd_color.r, backgrd_color.g,
                            backgrd_color.b, backgrd_color.a);
    SDL_RenderFillRect (renderer, &all);
  }

  display_misc_status ();

  // Draw the text indicating CPU numbers
  display_text ();

  // We fix the bounds of the "trace loop" so that the trace hovered by the
  // mouse pointer is displayed first
  int loop_start = 0;
  int loop_stop  = nb_traces;
  int loop_inc   = 1;

  if (nb_traces > 1 && mouse_in_gantt_zone) {
    if (point_inside_gantt (&mouse, 1)) {
      loop_start = nb_traces - 1;
      loop_stop  = -1;
      loop_inc   = -1;
    }
  }

  // main loop
  for (int _t = loop_start; _t != loop_stop; _t = _t + loop_inc) {
    trace_graphics_display_trace (_t, &selected);
  } // for (_t)

  // Mouse
  display_mouse_selection (&selected);

  SDL_RenderPresent (renderer);
}

// Control of view

void trace_graphics_toggle_vh_mode (void)
{
  horiz_mode ^= 1;

  if (tracking_mode) {
    tracking_mode = 0;
    SDL_SetTextureAlphaMod (track_tex, BUTTON_ALPHA);
  }
  if (footprint_mode) {
    footprint_mode = 0;
    SDL_SetTextureAlphaMod (footprint_tex, BUTTON_ALPHA);
  }
  trace_graphics_display ();
}

void trace_graphics_toggle_footprint_mode (void)
{
  static unsigned old_horiz, old_track;

  footprint_mode ^= 1;
  SDL_SetTextureAlphaMod (footprint_tex, footprint_mode ? 0xFF : BUTTON_ALPHA);

  if (footprint_mode) {
    old_horiz = horiz_mode;
    old_track = tracking_mode;
    horiz_mode = 1;
    tracking_mode = 0;
  } else {
    horiz_mode = old_horiz;
    tracking_mode = old_track;
  }
  SDL_SetTextureAlphaMod (track_tex, tracking_mode ? 0xFF : BUTTON_ALPHA);

  trace_graphics_display ();
}

void trace_graphics_toggle_tracking_mode (void)
{
  if (nb_traces > 1) {
    tracking_mode ^= 1;
    SDL_SetTextureAlphaMod (track_tex, tracking_mode ? 0xFF : BUTTON_ALPHA);

    horiz_mode = 0;

    if (footprint_mode) {
      footprint_mode = 0;
      SDL_SetTextureAlphaMod (footprint_tex, BUTTON_ALPHA);
    }

    trace_graphics_display ();
  } else
    printf ("Warning: tracking mode is only available when visualizing two "
            "traces\n");
}

static void trace_graphics_set_quick_nav (int nav)
{
  quick_nav_mode = nav;

  SDL_SetTextureAlphaMod (quick_nav_tex, quick_nav_mode ? 0xFF : BUTTON_ALPHA);
}

static void set_bounds (long start, long end)
{
  start_time = start;
  end_time   = end;
  duration   = end_time - start_time;

  for (int t = 0; t < nb_traces; t++) {
    trace_ctrl[t].first_displayed_iter =
        trace_data_search_next_iteration (&trace[t], start_time) + 1;
    trace_ctrl[t].last_displayed_iter =
        trace_data_search_prev_iteration (&trace[t], end_time) + 1;
  }
}

static void update_bounds (void)
{
  long start, end;
  int li[2];

  if (trace_ctrl[0].first_displayed_iter > trace[0].nb_iterations)
    start = iteration_start_time (
        trace + nb_traces - 1,
        trace_ctrl[nb_traces - 1].first_displayed_iter - 1);
  else if (trace_ctrl[nb_traces - 1].first_displayed_iter >
           trace[nb_traces - 1].nb_iterations)
    start =
        iteration_start_time (trace, trace_ctrl[0].first_displayed_iter - 1);
  else
    start = min (
        iteration_start_time (trace, trace_ctrl[0].first_displayed_iter - 1),
        iteration_start_time (trace + nb_traces - 1,
                              trace_ctrl[nb_traces - 1].first_displayed_iter -
                                  1));

  if (trace_ctrl[0].last_displayed_iter > trace[0].nb_iterations)
    li[0] = trace[0].nb_iterations - 1;
  else
    li[0] = trace_ctrl[0].last_displayed_iter - 1;

  if (trace_ctrl[nb_traces - 1].last_displayed_iter >
      trace[nb_traces - 1].nb_iterations)
    li[1] = trace[nb_traces - 1].nb_iterations - 1;
  else
    li[1] = trace_ctrl[nb_traces - 1].last_displayed_iter - 1;

  end = max (iteration_end_time (trace, li[0]),
             iteration_end_time (trace + nb_traces - 1, li[1]));

  set_bounds (start, end);
}

static void set_widest_iteration_range (int first, int last)
{
  trace_ctrl[0].first_displayed_iter             = first;
  trace_ctrl[nb_traces - 1].first_displayed_iter = first;

  trace_ctrl[0].last_displayed_iter             = last;
  trace_ctrl[nb_traces - 1].last_displayed_iter = last;

  update_bounds ();
}

static void set_iteration_range (int trace_num)
{
  if (nb_traces > 1) {
    int other = 1 - trace_num;

    trace_ctrl[other].first_displayed_iter =
        trace_ctrl[trace_num].first_displayed_iter;
    trace_ctrl[other].last_displayed_iter =
        trace_ctrl[trace_num].last_displayed_iter;
  }

  start_time = iteration_start_time (
      trace + trace_num, trace_ctrl[trace_num].first_displayed_iter - 1);
  end_time = iteration_end_time (trace + trace_num,
                                 trace_ctrl[trace_num].last_displayed_iter - 1);

  duration = end_time - start_time;
}

void trace_graphics_scroll (int delta)
{
  long start = start_time + duration * SHIFT_FACTOR * delta;
  long end   = start + duration;

  if (start < 0) {
    start = 0;
    end   = duration;
  }

  if (end > max_time) {
    end   = max_time;
    start = end - duration;
  }

  if (start != start_time || end != end_time) {
    set_bounds (start, end);
    trace_graphics_set_quick_nav (0);

    trace_graphics_display ();
  }
}

void trace_graphics_shift_left (void)
{
  if (quick_nav_mode) {
    int longest = (trace[0].nb_iterations >= trace[nb_traces - 1].nb_iterations)
                      ? 0
                      : nb_traces - 1;

    if (trace_ctrl[longest].last_displayed_iter < max_iterations) {
      trace_ctrl[longest].first_displayed_iter++;
      trace_ctrl[longest].last_displayed_iter++;

      set_iteration_range (longest);

      trace_graphics_display ();
    }
  } else {
    trace_graphics_scroll (1);
  }
}

void trace_graphics_shift_right (void)
{
  if (quick_nav_mode) {
    int longest = (trace[0].nb_iterations >= trace[nb_traces - 1].nb_iterations)
                      ? 0
                      : nb_traces - 1;

    if (trace_ctrl[longest].first_displayed_iter > 1) {
      trace_ctrl[longest].first_displayed_iter--;
      trace_ctrl[longest].last_displayed_iter--;

      set_iteration_range (longest);

      trace_graphics_display ();
    }
  } else {
    trace_graphics_scroll (-1);
  }
}

void trace_graphics_zoom_in (void)
{
  if (quick_nav_mode && (trace_ctrl[0].last_displayed_iter >
                         trace_ctrl[0].first_displayed_iter)) {

    int longest = (trace[0].nb_iterations >= trace[nb_traces - 1].nb_iterations)
                      ? 0
                      : nb_traces - 1;

    trace_ctrl[longest].last_displayed_iter--;

    set_iteration_range (longest);

    trace_graphics_display ();
  } else if (end_time > start_time + MIN_DURATION) {
    long start = start_time + duration * SHIFT_FACTOR;
    long end   = end_time - duration * SHIFT_FACTOR;

    if (end < start + MIN_DURATION)
      end = start + MIN_DURATION;

    set_bounds (start, end);
    trace_graphics_set_quick_nav (0);

    trace_graphics_display ();
  }
}

void trace_graphics_zoom_out (void)
{
  if (quick_nav_mode) {
    int longest = (trace[0].nb_iterations >= trace[nb_traces - 1].nb_iterations)
                      ? 0
                      : nb_traces - 1;

    if (trace_ctrl[longest].last_displayed_iter < max_iterations) {
      trace_ctrl[longest].last_displayed_iter++;

      set_iteration_range (longest);

      trace_graphics_display ();
    } else if (trace_ctrl[longest].first_displayed_iter > 1) {
      trace_ctrl[longest].first_displayed_iter--;

      set_iteration_range (longest);

      trace_graphics_display ();
    }
  } else {
    long start = start_time - duration * SHIFT_FACTOR;
    long end   = end_time + duration * SHIFT_FACTOR;

    if (start < 0)
      start = 0;

    if (end > max_time)
      end = max_time;

    if (start != start_time || end != end_time) {
      set_bounds (start, end);
      trace_graphics_set_quick_nav (0);

      trace_graphics_display ();
    }
  }
}

void trace_graphics_zoom_to_selection (void)
{
  if (selection_duration > 0) {
    long start = selection_start_time;
    long end   = selection_start_time + selection_duration;

    if (selection_duration < MIN_DURATION) {
      long delta = (MIN_DURATION - selection_duration) / 2;
      if (start < delta)
        start = 0;
      else
        start -= delta;
      end = start + MIN_DURATION;
      if (end > max_time) {
        end   = max_time;
        start = end - MIN_DURATION;
      }
    }

    set_bounds (start, end);
    trace_graphics_set_quick_nav (0);

    selection_duration = 0;

    trace_graphics_display ();
  }
}

void trace_graphics_mouse_moved (int x, int y)
{
  mouse.x = x;
  mouse.y = y;

  if (point_inside_gantts (&mouse))
    mouse_in_gantt_zone = 1;
  else
    mouse_in_gantt_zone = 0;

  if (mouse_down) {
    // Check if mouse in out-of-range on the x-axis
    if (point_in_yrange (&gantts_bounding_box, y)) {
      if (x < trace_display_info[0].gantt.x) {
        x = trace_display_info[0].gantt.x;
        trace_graphics_scroll (-1);
      } else if (x > trace_display_info[0].gantt.x +
                         trace_display_info[0].gantt.w - 1) {
        x = trace_display_info[0].gantt.x + trace_display_info[0].gantt.w - 1;
        trace_graphics_scroll (1);
      }
    }

    long new_pos = pixel_to_time (x);

    if (new_pos < mouse_orig_time) {
      selection_start_time = new_pos;
      selection_duration   = mouse_orig_time - new_pos;
    } else {
      selection_start_time = mouse_orig_time;
      selection_duration   = new_pos - mouse_orig_time;
    }
  }

  trace_graphics_display ();
}

void trace_graphics_mouse_down (int x, int y)
{
  mouse.x = x;
  mouse.y = y;

  if (point_inside_gantts (&mouse)) {

    mouse_orig_time    = pixel_to_time (x);
    selection_duration = 0;

    mouse_down = 1;

    trace_graphics_display ();
  }
}

void trace_graphics_mouse_up (int x, int y)
{
  mouse_down = 0;
}

void trace_graphics_setview (int first, int last)
{
  int last_disp_it, first_disp_it;

  // Check parameters and make sure iteration range is correct
  if (last < 1)
    last_disp_it = 1;
  else if (last > max_iterations)
    last_disp_it = max_iterations;
  else
    last_disp_it = last;

  if (first < 1)
    first_disp_it = 1;
  else if (first > last_disp_it)
    first_disp_it = last_disp_it;
  else
    first_disp_it = first;

  trace_graphics_set_quick_nav (trace_data_align_mode);

  set_widest_iteration_range (first_disp_it, last_disp_it);

  trace_graphics_display ();
}

void trace_graphics_reset_zoom (void)
{
  if (trace_data_align_mode) {

    if (!quick_nav_mode) {
      int first, last;

      if (trace_ctrl[0].first_displayed_iter >
          trace_ctrl[nb_traces - 1].last_displayed_iter)
        first = trace_ctrl[0].first_displayed_iter;
      else if (trace_ctrl[nb_traces - 1].first_displayed_iter >
               trace_ctrl[0].last_displayed_iter)
        first = trace_ctrl[nb_traces - 1].first_displayed_iter;
      else
        first = min (trace_ctrl[0].first_displayed_iter,
                     trace_ctrl[nb_traces - 1].first_displayed_iter);

      last = max (trace_ctrl[0].last_displayed_iter,
                  trace_ctrl[nb_traces - 1].last_displayed_iter);

      set_widest_iteration_range (first, last);

      trace_graphics_set_quick_nav (1);

      trace_graphics_display ();
    } else {
      trace_graphics_set_quick_nav (0);

      trace_graphics_display ();
    }
  }
}

void trace_graphics_display_all (void)
{
  set_widest_iteration_range (1, max_iterations);

  trace_graphics_set_quick_nav (trace_data_align_mode);

  trace_graphics_display ();
}

void trace_graphics_toggle_align_mode ()
{
  if (nb_traces == 1)
    return;

  trace_data_align_mode ^= 1;

  SDL_SetTextureAlphaMod (align_tex,
                          trace_data_align_mode ? 0xFF : BUTTON_ALPHA);

  max_time = max (iteration_end_time (trace, trace[0].nb_iterations - 1),
                  iteration_end_time (trace + nb_traces - 1,
                                      trace[nb_traces - 1].nb_iterations - 1));

  if (end_time > max_time) {
    end_time = max_time;
    duration = end_time - start_time;
  }

  set_bounds (start_time, end_time);
  trace_graphics_set_quick_nav (0);

  trace_graphics_display ();
}

void trace_graphics_relayout (unsigned w, unsigned h)
{
  WINDOW_WIDTH  = w;
  WINDOW_HEIGHT = h;

  layout_recompute ();
  layout_place_buttons ();

  create_cpu_textures (the_font);

  trace_graphics_display ();
}

void trace_graphics_init (unsigned w, unsigned h)
{
  max_iterations =
      max (trace[0].nb_iterations, trace[nb_traces - 1].nb_iterations);
  max_cores = max (trace[0].nb_cores, trace[nb_traces - 1].nb_cores);
  max_time  = max (iteration_end_time (trace, trace[0].nb_iterations - 1),
                  iteration_end_time (trace + nb_traces - 1,
                                      trace[nb_traces - 1].nb_iterations - 1));

  const unsigned min_width  = layout_get_min_width ();
  const unsigned min_height = layout_get_min_height ();

  WINDOW_WIDTH  = max (w, min_width);
  WINDOW_HEIGHT = max (h, min_height);

  WINDOW_HEIGHT = min (WINDOW_HEIGHT, layout_get_max_height ());

  layout_recompute ();

  if (SDL_Init (SDL_INIT_VIDEO) != 0)
    exit_with_error ("SDL_Init");

  char wintitle[1024];

  if (nb_traces == 1)
    sprintf (wintitle, "EasyView Trace Viewer -- \"%s\"", trace[0].label);
  else
    sprintf (wintitle, "EasyView -- \"%s\" (top) VS \"%s\" (bottom)",
             trace[0].label, trace[1].label);

  window = SDL_CreateWindow (wintitle, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
                             SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
  if (window == NULL)
    exit_with_error ("SDL_CreateWindow");

  SDL_SetWindowMinimumSize (window, min_width, min_height);

  // No Vsync provides a (much) smoother experience
  Uint32 render_flags = 0;

  if (soft_rendering)
    render_flags = SDL_RENDERER_SOFTWARE;
  else
    render_flags = SDL_RENDERER_ACCELERATED;

  renderer = SDL_CreateRenderer (window, -1, render_flags);
  if (renderer == NULL)
    exit_with_error ("SDL_CreateRenderer");

  SDL_RendererInfo info;
  SDL_GetRendererInfo (renderer, &info);
  printf ("Renderer used: [%s]\n", info.name);

  create_task_textures (min (max_cores, MAX_COLORS));

  create_misc_tex ();

  layout_place_buttons ();

  if (TTF_Init () < 0)
    exit_with_error ("TTF_Init");

  the_font = TTF_OpenFont ("fonts/FreeSansBold.ttf", FONT_HEIGHT - 4);

  if (the_font == NULL)
    exit_with_error ("TTF_OpenFont: %s", TTF_GetError ());

  create_text_texture (the_font);

#ifdef PRELOAD_THUMBNAILS
  preload_thumbnails (max_iterations);
#endif

  SDL_SetRenderDrawBlendMode (renderer, SDL_BLENDMODE_BLEND);
}
