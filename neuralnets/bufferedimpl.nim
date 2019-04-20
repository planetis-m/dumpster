const
   size* = 8192

type
   Buffer* = object
      data*: ptr array[size, char]
      len*: int

proc `=destroy`*(x: var Buffer) =
   if x.data != nil:
      dealloc(x.data)
      x.data = nil
      x.len = 0

proc `=sink`*(a: var Buffer; b: Buffer) =
   if a.data != nil and a.data != b.data:
      dealloc(a.data)
   a.len = b.len
   a.data = b.data

proc `=`*(a: var Buffer; b: Buffer) =
   if a.data != nil and a.data != b.data:
      dealloc(a.data)
      a.data = nil
   a.len = b.len
   if b.data != nil:
      a.data = cast[type(a.data)](alloc(size * sizeof(char)))
      copyMem(a.data, b.data, size * sizeof(char))

proc initBuffer*(): Buffer =
   result.data = cast[type(result.data)](alloc(size * sizeof(char)))
   result.len = 0

template `[]`*(x: Buffer; i: Natural): char =
   assert i < x.len
   x.data[i]

template `[]=`*(x: Buffer; i: Natural; y: char) =
   assert i < x.len
   x.data[i] = y
