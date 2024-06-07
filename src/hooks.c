#include "hooks.h"
#include "debug.h"
#include "error.h"
#include "global.h"
#include "ocl.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#define DLSYM_FLAG RTLD_SELF
#else
#define DLSYM_FLAG NULL
#endif

draw_func_t the_config      = NULL;
void_func_t the_init        = NULL;
void_func_t the_first_touch = NULL;
draw_func_t the_draw        = NULL;
void_func_t the_finalize    = NULL;
int_func_t the_compute      = NULL;
void_func_t the_refresh_img = NULL;
void_func_t the_tile_check  = NULL;

static tile_func_t the_tile_func = NULL;

void *hooks_find_symbol (char *symbol)
{
  return dlsym (DLSYM_FLAG, symbol);
}

static void *bind_it (char *kernel, char *s, char *variant, int print_error)
{
  char buffer[1024];
  void *fun = NULL;
  sprintf (buffer, "%s_%s_%s", kernel, s, variant);
  fun = hooks_find_symbol (buffer);
  if (fun != NULL)
    PRINT_DEBUG ('c', "Found [%s]\n", buffer);
  else {
    if (print_error == 2)
      exit_with_error ("Cannot resolve function [%s]", buffer);

    sprintf (buffer, "%s_%s", kernel, s);
    fun = hooks_find_symbol (buffer);

    if (fun != NULL)
      PRINT_DEBUG ('c', "Found [%s]\n", buffer);
    else if (print_error)
      exit_with_error ("Cannot resolve function [%s]", buffer);
  }
  return fun;
}

static void *bind_tile (char *kernel)
{
  char buffer[1024];
  void *fun = NULL;

  // First try to obey user
  if (tile_name != NULL) {
    sprintf (buffer, "%s_do_tile_%s", kernel, tile_name);
    fun = hooks_find_symbol (buffer);
    if (fun != NULL) {
      PRINT_DEBUG ('c', "Found requested tiling func [%s]\n", buffer);
      return fun;
    }
    // requested tile_name didn't work
    exit_with_error ("Cannot resolve function [%s]\n", buffer);
  }

  // Try to explore EASYPAP_TILEPREF environment variable
  if (tile_name == NULL) {
    char flavor[128];
    char *env = getenv ("EASYPAP_TILEPREF");
    if (env != NULL) {
      int index_env = 0, index = 0;
      for (;;) {
        if (env[index_env] == ':' ||
            env[index_env] == '\0') { // end of flavor name
          flavor[index] = '\0';
          sprintf (buffer, "%s_do_tile_%s", kernel, flavor);
          fun = hooks_find_symbol (buffer);
          if (fun != NULL) {
            PRINT_DEBUG ('c', "Found preferred tiling func [%s]\n", buffer);
            tile_name = malloc (strlen (flavor) + 1);
            strcpy (tile_name, flavor);
            return fun;
          }
          // flavor not found
          if (env[index_env] == '\0')
            break;
          index_env++;
          index = 0;
        }
        // regular character
        flavor[index++] = env[index_env++];
      }
    }
  }

  // Well, try default do_tile function
  sprintf (buffer, "%s_do_tile_default", kernel);
  fun = hooks_find_symbol (buffer);
  if (fun != NULL) {
    PRINT_DEBUG ('c', "Found [%s]\n", buffer);
    tile_name = "default";
    return fun;
  }

  // No tile function found
  tile_name = "none";
  return NULL;
}

void hooks_establish_bindings (int silent)
{
  if (opencl_used) {
    the_compute = bind_it (kernel_name, "invoke", variant_name, 0);
    if (the_compute == NULL) {
      the_compute = ocl_invoke_kernel_generic;
      PRINT_DEBUG ('c', "Using generic [%s] OpenCL kernel launcher\n",
                   "ocl_compute");
    }
  } else {
    the_compute = bind_it (kernel_name, "compute", variant_name, 2);
  }

  the_config      = bind_it (kernel_name, "config", variant_name, 0);
  the_init        = bind_it (kernel_name, "init", variant_name, 0);
  the_draw        = bind_it (kernel_name, "draw", variant_name, 0);
  the_finalize    = bind_it (kernel_name, "finalize", variant_name, 0);
  the_refresh_img = bind_it (kernel_name, "refresh_img", variant_name, 0);

  if (!opencl_used) {
    the_first_touch = bind_it (kernel_name, "ft", variant_name, do_first_touch);
  }

  the_tile_func  = bind_tile (kernel_name);
  the_tile_check = bind_it (kernel_name, "tile_check", tile_name, 0);

  if (!silent)
    PRINT_MASTER ("Using kernel [%s], variant [%s], tiling [%s]\n", kernel_name,
                  variant_name, tile_name);
}

void hooks_draw_helper (char *suffix, void_func_t default_func)
{
  char func_name[1024];
  void_func_t f = NULL;

  if (suffix == NULL)
    f = default_func;
  else {
    sprintf (func_name, "%s_draw_%s", kernel_name, suffix);
    f = hooks_find_symbol (func_name);

    if (f == NULL) {
      PRINT_DEBUG ('g', "Cannot resolve draw function: %s\n", func_name);
      f = default_func;
    }
  }

  f ();
}

int do_tile (int x, int y, int width, int height, int who)
{
  if (the_tile_func == NULL)
    exit_with_error ("No appropriate do_tile function found");

  monitoring_start_tile (who);

  int r = the_tile_func (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);

  return r;
}
