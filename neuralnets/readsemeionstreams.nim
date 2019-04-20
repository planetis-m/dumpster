import std / streams, strutils

type
   MatcherState = enum
      InputExpected, TargetExpected

const
   path = "semeion.data"
   nips = 256
   nops = 10

proc main =
   let file = newFileStream(path)
   var
      inputs = newSeq[float](nips)
      targets = newSeq[float](nops)

   var state = InputExpected
   var index = 0
   var count = 0
   while not file.atEnd:
      # Fill the weights array
      case state
      of InputExpected:
         inputs[index] = file.readFloat32()
         if index >= nips - 1:
            state = TargetExpected
            index = 0
         else:
            index.inc
      of TargetExpected:
         targets[index] = file.readFloat32()
         if index >= nops - 1:
            state = InputExpected
            index = 0
         else:
            index.inc
      # Skip whitespace chars
      while not file.atEnd and file.readChar in Whitespace:
         discard
      file.setPosition(file.getPosition - 1)
   close(file)

main()
