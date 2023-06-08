proc cond_swap(x, y: var int) =
  # var r = x < y
  # let tmp = if r: x else: y
  # y = if r: y else: x
  # x = tmp

  let tmp = min(x, y)
  y = max(x, y)
  x = tmp

proc partially_sorted_swap(x, y, z: var int) {.nosideEffect.} =
  # var r = z < x
  # var tmp = if r: z else: x
  # z = if r: x else: z
  # r = tmp < y
  # x = if r: x else: y
  # y = if r: y else: tmp

  let tmp = min(z, x)
  z = max(x, z)
  x =if tmp <= y: x else: y
  y = max(y, tmp)

proc sort3_maybe_branchless(x1, x2, x3: var int) {.nosideEffect.} =
  cond_swap(x2, x3)
  partially_sorted_swap(x1, x2, x3)

proc sort4_maybe_branchless(x1, x2, x3, x4: var int) {.nosideEffect.} =
  cond_swap(x1, x3)
  cond_swap(x2, x4)
  cond_swap(x1, x2)
  cond_swap(x3, x4)
  cond_swap(x2, x3)

proc sort5_maybe_branchless(x1, x2, x3, x4, x5: var int) {.nosideEffect.} =
  cond_swap(x1, x2)
  cond_swap(x4, x5)
  partially_sorted_swap(x3, x4, x5)
  cond_swap(x2, x5)
  partially_sorted_swap(x1, x3, x4)
  partially_sorted_swap(x2, x3, x4)

import std/strformat

proc main() =
  var arr1 = [5, -2, 9]
  sort3_maybe_branchless(arr1[0], arr1[1], arr1[2])
  echo &"Sorted array: {arr1[0]} {arr1[1]} {arr1[2]}"

  var arr2 = [-1, 8, 3, -6]
  sort4_maybe_branchless(arr2[0], arr2[1], arr2[2], arr2[3])
  echo &"Sorted array: {arr2[0]} {arr2[1]} {arr2[2]} {arr2[3]}"

  var arr3 = [2, -7, 10, 4, -3]
  sort5_maybe_branchless(arr3[0], arr3[1], arr3[2], arr3[3], arr3[4])
  echo &"Sorted array: {arr3[0]} {arr3[1]} {arr3[2]} {arr3[3]} {arr3[4]}"

main()
