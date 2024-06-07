#ifndef HOOKS_IS_DEF
#define HOOKS_IS_DEF

typedef void (*void_func_t) (void);
typedef unsigned (*int_func_t) (unsigned);
typedef void (*draw_func_t) (char *);
typedef int (*tile_func_t) (int, int, int, int);

extern draw_func_t the_config;
extern void_func_t the_init;
extern void_func_t the_first_touch;
extern draw_func_t the_draw;
extern void_func_t the_finalize;
extern int_func_t the_compute;
extern void_func_t the_refresh_img;
extern void_func_t the_tile_check;

void *hooks_find_symbol (char *symbol);
void hooks_establish_bindings (int silent);

// Call function ${kernel}_draw_${suffix}, or default_func if symbol not found
void hooks_draw_helper (char *suffix, void_func_t default_func);

// Call appropriate do_tile_${suffix} function, with calls to monitoring start/end
int do_tile (int x, int y, int width, int height, int who);

#endif
