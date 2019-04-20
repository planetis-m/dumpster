type
   IterState = enum
      Last, First, PastStart

   CountIter = object
      state: MatcherState
      v, a, b: int

proc hasNext(x: CountIter): bool {.inline.} =
   x.state >= First

proc init(x: var CountIter; a, b: int) =
   x.state = First
   x.a = a
   x.b = b
   x.v = x.a

proc next(x: var CountIter): int {.inline.} =
   if x.state == PastStart:
      x.v.inc
   else: x.state = PastStart
   if x.v >= x.b:
      x.state = Last
   result = x.v

iterator to(a, b: int): int =
   var x: CountIter
   x.init(a, b)
   while hasNext(x):
      yield x.next()

import times, strutils

template bench(name: string, code: untyped) =
   proc runBench() {.gensym.} =
      let start = epochTime()
      code
      let duration = epochTime() - start
      let timeStr = formatEng(duration)
      echo name, ": ", timeStr
   runBench()

proc main =
   bench("countup"):
      var y = 0
      for i in 1 .. 2000:
         y.inc
      doAssert y == 2000

   bench("CountIter"):
      var y = 0
      var x: CountIter
      x.init(1, 2000)
      while hasNext(x):
         discard x.next()
         y.inc
      doAssert y == 2000

   bench("CountIterForLoop"):
      var y = 0
      for i in 1.to 2000:
         y.inc
      doAssert y == 2000

main()
