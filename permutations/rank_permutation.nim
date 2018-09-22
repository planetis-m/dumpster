import math

type
   Iterable = concept c
      for x in items(c): discard

   Indexable[T] = concept c
      var i: int
      c[i] is T
      c.len is int

proc isPerm*(arr: Iterable): bool =
   let n = len(arr)
   var used = newSeq[bool](n)
   for a in arr:
      if a >= 0 and a < n and not used[a]:
         used[a] = true
      else:
         return false
   result = true

proc rank*(arr: Indexable[int]): int =
   assert(isPerm(arr), "arr is not a permutation")
   let h = high(arr)
   for i in 0 .. h - 1:
      let f = fac(h - i)
      for j in i + 1 .. h:
         if arr[j] < arr[i]:
            result += f

proc permutate*(arr: var Indexable[int], rank: int) =
   assert(isPerm(arr), "arr is not a permutation")
   var rank = rank
   let h = high(arr)
   for i in 0 .. h - 1:
      let f = fac(h - i)
      for j in i + 1 .. h:
         if rank - f >= 0:
            swap(arr[j], arr[i])
            rank -= f
         else:
            break

let a = @[0, 8, 4, 2, 7, 6, 5, 1, 3]
echo a.rank
