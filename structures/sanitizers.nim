# https://github.com/google/sanitizers/wiki/AddressSanitizerManualPoisoning

proc asan_poison_memory_region(region: pointer, size: int) {.
  header: "sanitizer/asan_interface.h", importc: "__asan_poison_memory_region".}
proc asan_unpoison_memory_region(region: pointer, size: int) {.
  header: "sanitizer/asan_interface.h", importc: "__asan_unpoison_memory_region".}

template poisonMemRegion*(region: pointer, size: int) =
  when defined(addrsan):
    asan_poison_memory_region(region, size)

template unpoisonMemRegion*(region: pointer, size: int) =
  when defined(addrsan):
    asan_unpoison_memory_region(region, size)
