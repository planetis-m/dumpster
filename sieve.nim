import math


proc eratosthenes(n: Natural): int =
   var p = newSeq[int8](n + 1)
   p[0] = 1
   p[1] = 1

   let slimit = sqrt(n.float).int
   for i in 2 .. slimit:
      if p[i] == 0:
         for j in countup(i * i, n, i):
            p[j] = 1

   for i in p:
      result.inc i
   result = n - result

proc eratosthenes2(n: Natural): int =
   let n = (n + 1) shr 1
   var
      p = newSeq[int8](n)
      i = 1
      j = 3

   for i in 0 .. p.high:
      p[i] = 1

   while i < n:
      if p[i] > 0:
         for k in countup(j * j shr 1, n - 1, i):
            p[k] = 0
      i.inc(1)
      j.inc(2)

   for i in p:
      result.inc i


echo eratosthenes2(1_000_000)
