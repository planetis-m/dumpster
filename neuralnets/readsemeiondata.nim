import std / parseutils, strutils
import fileBuffered

type
   MatcherState = enum
      InputExpected, TargetExpected

const
   path = "semeion.data"
   nips = 256
   nops = 10

proc main =
   let file = open(path)
   var
      inputs = newSeq[float](nips)
      targets = newSeq[float](nops)

   var state = InputExpected
   var index = 0
   for len, chunk in file.buffered:
      # Fill the weights array
      var chunkPos = 0
      while true:
         # Skip whitespace chars
         while chunkPos < len and chunk[chunkPos] in Whitespace:
            chunkPos.inc
         if chunkPos >= len: break
         # Determine the correct array
         var bytesParsed = 0
         case state
         of InputExpected:
            bytesParsed = chunk.parseFloat(inputs[index], chunkPos)
            if index >= nips - 1:
               state = TargetExpected
               index = 0
            else:
               index.inc
         of TargetExpected:
            bytesParsed = chunk.parseFloat(targets[index], chunkPos)
            if index >= nops - 1:
               state = InputExpected
               index = 0
            else:
               index.inc
         assert bytesParsed > 0
         chunkPos += bytesParsed

   close(file)

main()
