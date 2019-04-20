{.passC: "-march=native -ffast-math -fno-unroll-loops -fopt-info -fverbose-asm -masm=intel -S".}

const
   arraySize = 10000000
   memAlign = 16

type
   LArray = ptr array[arraySize, float]
   LObject = object
      p: pointer
      data: LArray

# template createLArray(): untyped = cast[LArray](alloc(arraySize * sizeof(float)))

proc createArray(): LObject =
   result.p = alloc(arraySize * sizeof(float) + memAlign - 1)
   let address = cast[ByteAddress](result.p)
   if (address and (memAlign - 1)) == 0:
      result.data = cast[LArray](address)
   else:
      let offset = memAlign - (address and (memAlign - 1))
      result.data = cast[LArray](address +% offset)

proc `=destroy`(x: var LObject) =
   if x.p != nil:
      dealloc(x.p)
      x.p = nil
      x.data = nil

proc `=sink`(a: var LObject; b: LObject) =
   if a.p != nil and a.p != b.p:
      dealloc(a.p)
   a.p = b.p
   a.data = b.data

# proc `=`*(a: var LObject; b: LObject) =
#    if a.p != nil and a.p != b.p:
#       dealloc(a.p)
#       a.p = nil
#       a.data = nil
#    if b.p != nil:
#       a.p = alloc(arraySize * sizeof(float) + memAlign - 1)
#       let address = cast[ByteAddress](a.p)
#       if (address and (memAlign - 1)) == 0:
#          a.data = cast[LArray](address)
#          copyMem(a.data, b.data, arraySize * sizeof(float))
#       else:
#          let offset = memAlign - (address and (memAlign - 1))
#          a.data = cast[LArray](address +% offset)
#          copyMem(a.data, b.data, arraySize * sizeof(float))

template restrict(arg, store: untyped) =
   let arg {.codegenDecl: "$# __restrict__ $#".} = store

proc dot(a, b: LObject): LObject =
   result = createArray()

   restrict(aptr, a.data)
   restrict(bptr, b.data)
   restrict(cptr, result.data)

   for j in 0 ..< 200: # some repetitions
      for i in 0 ..< arraySize:
         cptr[i] = aptr[i] * bptr[i]

proc main =
   var a = createArray()
   var b = createArray()

   var c = a.dot(b)

main()
