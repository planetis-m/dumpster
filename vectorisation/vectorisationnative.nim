# A Brief Introduction to Vectorization https://www.youtube.com/watch?v=4_trCnLQllw
# https://devsector.wordpress.com/2017/06/25/cpp-vectorization-diagnostics/
{.passC: "-march=native -ffast-math -fno-unroll-loops ".}
when defined(gcc):
   {.passC: "-fno-loop-interchange -fopt-info".}
elif defined(clang):
   {.passC: "Rpass=loop-vectorize -Rpass-analysis=loop-vectorize".}
when defined(assembly):
   {.passC: "-fverbose-asm -masm=intel -S".}
{.pragma: restrict, codegenDecl: "$# __restrict__ $#".}

const
   arraySize = 10000000

type
   LArray = ptr array[arraySize, float]

# template createArray(): untyped =
#    cast[LArray](alloc(arraySize * sizeof(float)))

template createArray(): LArray =
   let res {.restrict.} = cast[LArray](alloc(arraySize * sizeof(float)))
   res

template restrict(arg: untyped) =
   let arg {.codegenDecl: "$# __restrict__ $#".} = arg

proc dot(a, b: LArray): LArray =
   result = createArray()

   restrict(a)
   restrict(b)

   for j in 0 ..< 200: # some repetitions
      for i in 0 ..< arraySize:
         result[i] = a[i] * b[i]

proc main =
   var a = createArray()
   var b = createArray()

   var c = a.dot(b)
   dealloc(a)
   dealloc(b)
   dealloc(c)

main()
