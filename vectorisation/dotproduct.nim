# Also use -Wno-discarded-qualifiers -fopt-info
when defined(debugAsm):
   {.passC: "-fverbose-asm -masm=intel -S".}
when defined(fastmath):
   {.passC: "-ffast-math".}
when defined(marchNative):
   {.passC: "-march=native".}

import random

const
   n = 16384 div sizeof(float32) # L1 DCache size
   numTrials = 100

proc generateVector(n: int): seq[float32] =
   result = newSeq[float32](n)
   for i in 0 ..< n:
      result[i] = rand(1.0)

proc dotProduct(listA, listB: openarray[float32]): float32 =
   #
   # listA: a list of numbers
   # listB: a list of numbers of the same length as listA
   #
   let m = len(listA)
   for i in 0 ..< m:
      result += listA[i] * listB[i]

proc main() =
   echo "Generating ", n, " element vectors."
   let
      listA = generateVector(n)
      listB = generateVector(n)

   echo dotProduct(listA, listB)

main()
