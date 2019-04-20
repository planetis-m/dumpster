const
   size = 8192

type
   Buffer* = object
      data*: ptr array[size, char]
      len*: int

var
   allocCount, deallocCount: int

proc `=destroy`*(x: var Buffer) =
   if x.data != nil:
      dealloc(x.data)
      inc deallocCount
      x.data = nil
      x.len = 0

proc `=sink`*(a: var Buffer; b: Buffer) =
   if a.data != nil and a.data != b.data:
      dealloc(a.data)
      inc deallocCount
   a.len = b.len
   a.data = b.data

proc `=`*(a: var Buffer; b: Buffer) =
   if a.data != nil and a.data != b.data:
      dealloc(a.data)
      inc deallocCount
      a.data = nil
   a.len = b.len
   if b.data != nil:
      a.data = cast[type(a.data)](alloc(size * sizeof(char)))
      inc allocCount
      copyMem(a.data, b.data, size)

proc initBuffer*(): Buffer =
   result.data = cast[type(result.data)](alloc(size * sizeof(char)))
   inc allocCount
   result.len = size

from strutils import Whitespace

type
   BufferState = enum
      Last, First, PastStart

   FileBuffer* = object
      state: BufferState
      file: File
      start, bytesRead: int
      seps: set[char]
      v: Buffer

proc hasNext*(x: FileBuffer): bool =
   x.state >= First

proc init*(x: var FileBuffer; path: string; seps = Whitespace) =
   x.state = First
   x.file = open(path)
   x.v = initBuffer()
   x.seps = seps

proc next*(x: var FileBuffer): lent Buffer =
   if x.state == PastStart:
      # shift trimmed chars to the front
      var trimmedStart = x.v.len
      var trimmedLen = size - x.v.len
      x.start = 0
      while x.start < trimmedLen:
         x.v.data[x.start] = x.v.data[trimmedStart]
         trimmedStart.inc
         x.start.inc
   else: x.state = PastStart
   # Read next chunk
   x.bytesRead = readBuffer(x.file, addr(x.v.data[x.start]), size - x.start)
   # Trim partially read content
   x.v.len = x.start + x.bytesRead
   while x.v.len >= 0 and x.v.data[x.v.len - 1] notin x.seps:
      x.v.len.dec
   # Break iff buffer is only half-full
   if x.bytesRead < size - x.start:
      x.state = Last
   # Yield the buffer
   result = x.v

proc close*(x: FileBuffer) =
   close(x.file)

proc main =
   const path = "semeion.data"
   var x: FileBuffer
   x.init(path)

   while hasNext(x):
      echo x.next().len

   close(x)

main()
echo "after ", allocCount, " ", deallocCount
