type
   Row* = object
      inputs*: ptr array[256, float]
      targets*: ptr array[10, int]

template createArray[T](x): untyped =
   x = cast[type(x)](alloc(x[].len * sizeof(T)))

template destroyArray(x): untyped =
   if x != nil:
      dealloc(x)
      x = nil

template sinkArray(a, b): untyped =
   if a != nil and a != b:
      dealloc(a)
   a = b

template assignArray[T](a, b): untyped =
   if a != nil and a != b:
      dealloc(a)
      a = nil
   if b != nil:
      createArray[T](a)
      copyMem(a, b, sizeof(b))

proc `=destroy`*(x: var Row) =
   destroyArray(x.inputs)
   destroyArray(x.targets)

proc `=sink`*(a: var Row; b: Row) =
   sinkArray(a.inputs, b.inputs)
   sinkArray(a.targets, b.targets)

proc `=`*(a: var Row; b: Row) =
   assignArray[float](a.inputs, b.inputs)
   assignArray[int](a.targets, b.targets)

proc initRow*(): Row =
   createArray[float](result.inputs)
   createArray[int](result.targets)
