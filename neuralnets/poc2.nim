from strutils import Whitespace

type
   BufferState = enum
      Last, First, PastStart

   FileBuffer* = object
      state: BufferState
      file: File
      start, bytesRead: int
      seps: set[char]
      v: Chunk

   Chunk* = object
      buffer*: string
      len*: int

const
   bufferSize = 8192

proc hasNext*(x: FileBuffer): bool =
   x.state >= First

proc init*(x: var FileBuffer; path: string; seps = Whitespace) =
   x.state = First
   x.file = open(path)
   x.v.buffer = newString(bufferSize)
   x.seps = seps

proc next*(x: var FileBuffer): Chunk =
   if x.state == PastStart:
      # shift trimmed chars to the front
      var trimmedStart = x.v.len
      var trimmedLen = bufferSize - x.v.len
      x.start = 0
      while x.start < trimmedLen:
         x.v.buffer[x.start] = x.v.buffer[trimmedStart]
         trimmedStart.inc
         x.start.inc
   else: x.state = PastStart
   # Read next chunk
   x.bytesRead = readBuffer(x.file, addr(x.v.buffer[x.start]), bufferSize - x.start)
   # Trim partially read content
   x.v.len = x.start + x.bytesRead
   while x.v.len >= 0 and x.v.buffer[x.v.len - 1] notin x.seps:
      x.v.len.dec
   # Break iff buffer is only half-full
   if x.bytesRead < bufferSize - x.start:
      x.state = Last
   # Yield the buffer
   result = x.v

proc close*(x: FileBuffer) =
   close(x.file)

when isMainModule:
   proc main =
      const path = "semeion.data"
      var x: FileBuffer
      x.init(path)

      while hasNext(x):
         echo x.next().len

      close(x)

   main()
