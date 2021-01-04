import times, stats, strformat, macros

proc printStats(name: string, stats: RunningStat, dur: float) =
   echo &"""{name}:
   Collected {stats.n} samples in {dur:.4} seconds
   Average time: {stats.mean * 1000:.4} ms
   Stddev  time: {stats.standardDeviationS * 1000:.4} ms
   Min     time: {stats.min * 1000:.4} ms
   Max     time: {stats.max * 1000:.4} ms"""

template makeBench(name, samples, init, code: untyped) =
   proc runBench() {.gensym, nimcall.} =
      var stats: RunningStat
      init
      let globalStart = cpuTime()
      for i in 1 .. samples:
         let start = cpuTime()
         code
         let duration = cpuTime() - start
         stats.push duration
      let globalDuration = cpuTime() - globalStart
      printStats(name, stats, globalDuration)
   runBench()

# macro bench*(name: string; samples: int; metabody: untyped): untyped =
#    var init, code: NimNode
# #    # for init
# #    if body[0].kind in nnkCallKinds:
# #       if eqIdent(body[0][0], "init"):
# #          init = body[0][1]
# #          code = body[1][1]
# #       else:
# #          init = newNimNode(nnkStmtList)
# #          code = body[0]
# #    else:
# #       init = newNimNode(nnkStmtList)
# #       code = body[0]
#
#    for body in metabody:
#       body.expectKind nnkCall
#       body.expectLen 2
#       if eqIdent(body[0], "init"):
#          init = body[1]
#       elif eqIdent(body[0], "code"):
#          code = body[1]
#       else:
#          error("unexpectecd section", body[0])
#    result = getAst(makeBench(name, samples, init, code))
