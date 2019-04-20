import std / parseutils, strutils
import fileBuffered

type
   MatcherState = enum
      RowExpected, NewLineExpected

const
   separator = ','
   path = "iris.data"
   ncol = 5

proc main =
   let file = open(path)
   var row = newSeq[string](ncol)

   var state = RowExpected
   var index = 0
   for len, chunk in file.buffered:
      # Fill the weights array
      var chunkPos = 0
      while true:
         # Skip whitespace chars
         while chunkPos < len and chunk[chunkPos] == separator:
            chunkPos.inc
         if chunkPos >= len:
            break
         case state
         of RowExpected:
            let bytesParsed = chunk.parseUntil(row[index], NewLines+{separator}, chunkPos)
            assert bytesParsed > 0
            chunkPos += bytesParsed
            if index >= ncol - 1:
               state = NewLineExpected
               index = 0
            else:
               index.inc
         of NewLineExpected:
            echo row
            while chunkPos < len and chunk[chunkPos] in NewLines:
               chunkPos.inc
            state = RowExpected

   close(file)

main()
