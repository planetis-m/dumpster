import std / parseutils, strutils
import poc2

type
   ParserState = enum
      ReadRow, InputExpected, TargetExpected

   CsvParser* = object
      state: ParserState
      separator: char
      stream: FileBuffer
      chunk: Chunk
      index, chunkPos: int
      v: Row

   Row* = object
      inputs*: array[256, float]
      targets*: array[10, int]

proc hasNext*(x: CsvParser): bool =
   x.state >= InputExpected

proc init*(x: var CsvParser; path: string; separator = ' ') =
   x.stream.init(path)
   x.state = InputExpected
   x.separator = separator

proc next*(x: var CsvParser): Row =
   while true:
      # Skip whitespace chars
      while x.chunkPos < x.chunk.len and x.chunk.buffer[x.chunkPos] in NewLines+{x.separator}:
         x.chunkPos.inc

      if x.chunkPos >= x.chunk.len:
         if hasNext(x.stream):
            x.chunkPos = 0
            x.chunk = x.stream.next()

      case x.state
      of InputExpected:
         # Parse inputs
         let bytesParsed = x.chunk.buffer.parseFloat(x.v.inputs[x.index], x.chunkPos)
         assert bytesParsed > 0
         x.chunkPos += bytesParsed
         if x.index >= x.v.inputs.high:
            x.index = 0
            x.state = TargetExpected
         else:
            x.index.inc
      of TargetExpected:
         # Parse targets
         let bytesParsed = x.chunk.buffer.parseInt(x.v.targets[x.index], x.chunkPos)
         assert bytesParsed > 0
         x.chunkPos += bytesParsed
         if x.index >= x.v.targets.high:
            x.index = 0
            x.state = ReadRow
         else:
            x.index.inc
      of ReadRow:
         if x.chunkPos < x.chunk.len:
            x.state = InputExpected
         result = x.v
         break

proc close*(x: CsvParser) =
   close(x.stream)

when isMainModule:
   proc main =
      const path = "semeion.data"
      var x: CsvParser
      x.init(path)

      while hasNext(x):
         echo x.next().targets

      close(x)

   main()
