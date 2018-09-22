import math, random

const
   n = 100_000
   count = 120

type
   Iterable = concept c
      for x in items(c): discard

   Indexable[T] = concept c
      var i: int
      c[i] is T
      c.len is int

proc rank(arr: Indexable[int]): int =
   let h = high(arr)
   for i in 0 .. h - 1:
      let f = fac(h - i)
      for j in i + 1 .. h:
         if arr[j] < arr[i]:
            result += f

proc isRandom(freqs: Indexable[int]; n: int): bool =
   ## Calculates the chi-square value for N positive integers less than r
   ## Source: "Algorithms in C" - Robert Sedgewick - pp. 517
   ## NB: Sedgewick recommends: "...to be sure, the test should be tried a few times,
   ## since it could be wrong in about one out of ten times."
   let r = freqs.len
   let n_r = n/r
   # This is valid if N is greater than about 10r
   assert(n > 10 * r)
   # Calculate chi-square
   var chiSquare = 0.0
   for v in freqs:
      let f = float(v) - n_r
      chiSquare += pow(f, 2.0)
   chiSquare = chiSquare / n_r
   # The statistic should be within 2(r)^1/2 of r
   abs(chiSquare - float(r)) <= 2.0 * sqrt(float(r))

block randInt:
   # Get frequency of randoms
   var freqs: array[count, int]
   for i in 1 .. n:
      freqs[rand(count - 1)].inc
   doAssert isRandom(freqs, n)

block randFloat:
   var freqs: array[count, int]
   for i in 1 .. n:
      freqs[int(rand(1.0) * float(count))].inc
   doAssert isRandom(freqs, n)

block randShuffle:
   var freqs: array[count, int]
   for i in 1 .. n:
      var a = [0, 1, 2, 3, 4]
      shuffle(a)
      freqs[rank(a)].inc
   doAssert isRandom(freqs, n)
