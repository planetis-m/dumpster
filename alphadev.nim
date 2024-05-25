# // Ensures that __c(*__x, *__y) is true by swapping *__x and *__y if necessary.
# inline void cond_swap(int* x, int* y, bool (*c)(int, int)) {
#   bool r = c(*x, *y);
#   int tmp = r ? *x : *y;
#   *y = r ? *y : *x;
#   *x = tmp;
# }

# // Ensures that *__x, *__y and *__z are ordered according to the comparator __c,
# // under the assumption that *__y and *__z are already ordered.
# inline void partially_sorted_swap(int* x, int* y, int* z, bool (*c)(int, int)) {
#   bool r = c(*z, *x);
#   int tmp = r ? *z : *x;
#   *z = r ? *x : *z;
#   r = c(tmp, *y);
#   *x = r ? *x : *y;
#   *y = r ? *y : tmp;
# }

# void sort3_maybe_branchless(int* x1, int* x2, int* x3,
#                          bool (*c)(int, int)) {
#   cond_swap(x2, x3, c);
#   partially_sorted_swap(x1, x2, x3, c);
# }

# void sort4_maybe_branchless(int* x1, int* x2, int* x3,
#                          int* x4, bool (*c)(int, int)) {
#   cond_swap(x1, x3, c);
#   cond_swap(x2, x4, c);
#   cond_swap(x1, x2, c);
#   cond_swap(x3, x4, c);
#   cond_swap(x2, x3, c);
# }

# void sort5_maybe_branchless(int* x1, int* x2, int* x3,
#                          int* x4, int* x5, bool (*c)(int, int)) {
#   cond_swap(x1, x2, c);
#   cond_swap(x4, x5, c);
#   partially_sorted_swap(x3, x4, x5, c);
#   cond_swap(x2, x5, c);
#   partially_sorted_swap(x1, x3, x4, c);
#   partially_sorted_swap(x2, x3, x4, c);
# }

# bool compare(int a, int b) {
#     return a < b;
# }

proc cond_swap(x, y: var int, c: proc (a, b: int): bool {.nimcall.}) {.inline.} =
  var r = c(x, y)
  # let tmp = if r: x else: y
  var tmp: int
  {.emit: "`tmp` = `r` ? *`x` : *`y`;".}
  # y = if r: y else: x
  {.emit: "*`y` = `r` ? *`y` : *`x`;".}
  x = tmp

proc partially_sorted_swap(x, y, z: var int, c: proc (a, b: int): bool {.nimcall.}) {.inline.} =
  var r = c(z, x)
  # var tmp = if r: z else: x
  var tmp: int
  {.emit: "`tmp` = `r` ? *`z` : *`x`;".}
  # z = if r: x else: z
  {.emit: "*`z` = `r` ? *`x` : *`z`;".}
  r = c(tmp, y)
  # x = if r: x else: y
  {.emit: "*`x` = `r` ? *`x` : *`y`;".}
  # y = if r: y else: tmp
  {.emit: "*`y` = `r` ? *`y` : `tmp`;".}

proc sort3_maybe_branchless(x1, x2, x3: var int, c: proc (a, b: int): bool {.nimcall.}) =
  cond_swap(x2, x3, c)
  partially_sorted_swap(x1, x2, x3, c)

proc sort4_maybe_branchless(x1, x2, x3, x4: var int, c: proc (a, b: int): bool {.nimcall.}) =
  cond_swap(x1, x3, c)
  cond_swap(x2, x4, c)
  cond_swap(x1, x2, c)
  cond_swap(x3, x4, c)
  cond_swap(x2, x3, c)

proc sort5_maybe_branchless(x1, x2, x3, x4, x5: var int, c: proc (a, b: int): bool {.nimcall.}) =
  cond_swap(x1, x2, c)
  cond_swap(x4, x5, c)
  partially_sorted_swap(x3, x4, x5, c)
  cond_swap(x2, x5, c)
  partially_sorted_swap(x1, x3, x4, c)
  partially_sorted_swap(x2, x3, x4, c)

proc compare(a: int, b: int): bool = a < b

import std/strformat

proc main() =
  var arr1 = [5, -2, 9]
  sort3_maybe_branchless(arr1[0], arr1[1], arr1[2], compare)
  echo &"Sorted array: {arr1[0]} {arr1[1]} {arr1[2]}"

  var arr2 = [-1, 8, 3, -6]
  sort4_maybe_branchless(arr2[0], arr2[1], arr2[2], arr2[3], compare)
  echo &"Sorted array: {arr2[0]} {arr2[1]} {arr2[2]} {arr2[3]}"

  var arr3 = [2, -7, 10, 4, -3]
  sort5_maybe_branchless(arr3[0], arr3[1], arr3[2], arr3[3], arr3[4], compare)
  echo &"Sorted array: {arr3[0]} {arr3[1]} {arr3[2]} {arr3[3]} {arr3[4]}"

main()
