/* RTLD_NEXT probe for tests/async/host/test_shim_driver.c: the first library of the pair. Its
 * dlsym(RTLD_NEXT) call has to find libnext_b's definition even with the shim's dlsym hook in
 * between (the hook tail-calls glibc's dlsym, so glibc still sees this library as the caller). */
#define _GNU_SOURCE
#include <dlfcn.h>

int wg_next_probe(void) { return 1; }

int wg_call_next(void) {
  int (*f)(void) = (int (*)(void))dlsym(RTLD_NEXT, "wg_next_probe");
  return f ? f() : -1;
}
