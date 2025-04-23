const
   Size = 101

proc dot(a, b: seq[float]): seq[float] =
   result = newSeq[float](Size)
   for j in 0 ..< 2_000_000: # some repetitions
      for i in 0 ..< Size:
         result[i] = a[i] * b[i]

proc main =
   var a = newSeq[float](Size)
   var b = newSeq[float](Size)

   var c = a.dot(b)

main()
