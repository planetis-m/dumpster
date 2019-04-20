import std / parseutils, strutils
import fileBuffered

type
   MatcherState = enum
      InputExpected, TargetExpected

   Data = object
      inp: seq[seq[float]]   ## 2D floating point array of input.
      tg: seq[seq[float]]    ## 2D floating point array of target.

const
   path = "semeion.data"
   nips = 256
   nops = 10

proc initData: Data =
   ## Parses file from path getting all inputs and outputs for the neural network. Returns data object.
   let file = open(path, fmRead)

   var inps = newSeqOfCap[float](nips)
   var tgs = newSeqOfCap[float](nops)
   for line in file.lines:
      var col = 0
      for str in line.splitWhitespace:
         let val = str.parseFloat()
         if col < nips: inps.add(val)
         else: tgs.add(val)
         col.inc
      result.inp.add inps
      result.tg.add tgs
      setLen(inps, 0)
      setLen(tgs, 0)
   close(file)

proc initDataBuffered: Data =
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
         # Parse float data
         var bytesParsed = 0
         case state
         of InputExpected:
            bytesParsed = chunk.parseFloat(inputs[index], chunkPos)
            if index >= nips - 1:
               state = TargetExpected
               result.inp.add inputs
               index = 0
            else:
               index.inc
         of TargetExpected:
            bytesParsed = chunk.parseFloat(targets[index], chunkPos)
            if index >= nops - 1:
               state = InputExpected
               result.tg.add targets
               index = 0
            else:
               index.inc
         assert bytesParsed > 0
         chunkPos += bytesParsed
   close(file)

import times, strutils, random

template bench*(name: string, code: untyped) =
   proc runBench() {.gensym.} =
      let start = epochTime()
      code
      let duration = epochTime() - start
      let timeStr = formatFloat(duration, ffDecimal, 3)
      echo name, ": ", timeStr
   runBench()

# proc main =
#    bench("readBuffered"):
#       for re in 1 .. 100:
#          discard initDataBuffered()
# 
#    bench("readLines"):
#       for re in 1 .. 100:
#          discard initData()
# 
# main()
proc test =
   let got = initDataBuffered()
   let expected = initData()

   assert got.inp.len == expected.inp.len
   assert got.tg.len == expected.tg.len
   for i in 0 ..< got.tg.len:
      assert got.tg[i] == expected.tg[i]
   for i in 0 ..< got.inp.len:
      assert got.inp[i] == expected.inp[i]

test()
