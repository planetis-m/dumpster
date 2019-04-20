{.passC: "-march=native -ffast-math".}
when defined(loopvec):
   {.passC: "-Rpass=loop-vectorize -Rpass-analysis=loop-vectorize -Rpass-missed=loop-vectorize".}
elif defined(slpvec):
   {.passC: "-Rpass=slp-vectorizer -Rpass-analysis=slp-vectorizer -Rpass-missed=slp-vectorizer".}
when defined(assembly): {.passC: "-fverbose-asm -masm=intel -S".}
type
   MyArray = UncheckedArray[float]

const size = 100
template createArray(size: int): untyped = cast[ptr MyArray](alloc(size * sizeof(float)))

proc dot(a, b: ptr MyArray): ptr MyArray =
   result = createArray(size)
   for i in 0 ..< size:
      result[i] = a[i] * b[i]

proc main =
   var a = createArray(size)
   var b = createArray(size)
   var c = a.dot(b)

   dealloc(a)
   dealloc(b)
   dealloc(c)

main()
