from strutils import Whitespace
import bufferedimpl

type
   StreamState = enum
      Last, First, PastStart

   FileStream* = object
      state: StreamState
      file: File
      start, bytesRead: int
      seps: set[char]
      v: Buffer

proc hasNext*(x: FileStream): bool =
   x.state >= First

proc init*(x: var FileStream; path: string; seps = Whitespace) =
   x.state = First
   x.file = open(path)
   x.v = initBuffer()
   x.seps = seps

proc next*(x: var FileStream): lent Buffer =
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

proc close*(x: FileStream) =
   close(x.file)

proc main =
   const path = "semeion.data"
   var x: FileStream
   x.init(path)

   while hasNext(x):
      echo x.next().len

   close(x)

main()
