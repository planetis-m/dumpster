when defined(assembly): {.passC: "-fverbose-asm -masm=intel -S".}
{.passC: "-march=native -ffast-math".}
{.passC: "-Rpass=loop-vectorize -Rpass-analysis=loop-vectorize -Rpass-missed=loop-vectorize".}
import math
const
   arraySize = 1000

type
   MyArray = ptr array[arraySize, float]

template createArray(): untyped =
   cast[MyArray](alloc(arraySize * sizeof(float)))

template restrict(arg: untyped) =
   let arg {.codegenDecl: "$# __restrict__ $#".} = arg

proc quad(a, b, c: MyArray, x1, x2: var MyArray) =
   restrict(x1)
   restrict(x2)
   for i in 0 ..< arraySize:
      var s = b[i] * b[i] - 4 * a[i] * c[i]
      if s >= 0:
         s = sqrt(s)
         x2[i] = (-b[i] + s) / (2.0 * a[i])
         x1[i] = (-b[i] - s) / (2.0 * a[i])
      else:
         x2[i] = 0.0
         x1[i] = 0.0

proc main =
   let a, b, c = createArray()
   var x1, x2 = createArray()

   quad(a, b, c, x1, x2)

   dealloc(a)
   dealloc(b)
   dealloc(c)
   dealloc(x1)
   dealloc(x2)

main()
