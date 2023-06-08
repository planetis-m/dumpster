func cond_swap[T](x, y: var T) =
  # if y <= x: swap(x, y)
  let tmp = min(x, y)
  y = max(x, y)
  x = tmp

func partially_sorted_swap[T](x, y, z: var T) =
  let tmp = min(z, x)
  z = max(x, z)
  x = if tmp <= y: x else: y
  # {.emit: ["*", x, " = ", tmp <= y, " ? *", x, " : *", y, ";"].}
  y = max(y, tmp)

func sort3_maybe_branchless[T](x1, x2, x3: var T) =
  cond_swap(x2, x3)
  partially_sorted_swap(x1, x2, x3)

func sort4_maybe_branchless[T](x1, x2, x3, x4: var T) =
  cond_swap(x1, x3)
  cond_swap(x2, x4)
  cond_swap(x1, x2)
  cond_swap(x3, x4)
  cond_swap(x2, x3)

func sort5_maybe_branchless[T](x1, x2, x3, x4, x5: var T) =
  cond_swap(x1, x2)
  cond_swap(x4, x5)
  partially_sorted_swap(x3, x4, x5)
  cond_swap(x2, x5)
  partially_sorted_swap(x1, x3, x4)
  partially_sorted_swap(x2, x3, x4)

when defined(runFuzzTests):
  proc insertionSort[T](s: var openarray[T]) =
    for i in 1 ..< len(s):
      let x = s[i]
      var j = i - 1
      while j >= 0 and s[j] > x:
        s[j + 1] = s[j]
        dec(j)
      s[j + 1] = x

  proc isPermutation[N: static[int], T](x, y: array[N, T]): bool =
    # First one is sorted, second one is not
    var y = y
    insertionSort(y)
    result = x == y

  import std/[random, algorithm]

  proc main =
    var arr1: array[3, int]
    var arr2: array[4, int]
    var arr3: array[5, int]

    for x in mitems(arr1):
      x = rand(-100..100)
    for x in mitems(arr2):
      x = rand(-100..100)
    for x in mitems(arr3):
      x = rand(-100..100)

    for i in 1 .. 1000:
      shuffle(arr1)
      shuffle(arr2)
      shuffle(arr3)

      var arr1_copy = arr1
      var arr2_copy = arr2
      var arr3_copy = arr3

      sort3_maybe_branchless(arr1_copy[0], arr1_copy[1], arr1_copy[2])
      sort4_maybe_branchless(arr2_copy[0], arr2_copy[1], arr2_copy[2], arr2_copy[3])
      sort5_maybe_branchless(arr3_copy[0], arr3_copy[1], arr3_copy[2], arr3_copy[3], arr3_copy[4])

      doassert isSorted(arr1_copy)
      doassert isSorted(arr2_copy)
      doassert isSorted(arr3_copy)

      doassert isPermutation(arr1_copy, arr1)
      doassert isPermutation(arr2_copy, arr2)
      doassert isPermutation(arr3_copy, arr3)

  main()
else:
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
