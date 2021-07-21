import cowstrings, sync/spsc_queue

const
  numIters = 1000000

var
  pong: Thread[void]
  q1: SpscQueue[String]
  q2: SpscQueue[String]

template pushLoop(tx, data: typed, body: untyped): untyped =
  while not tx.tryPush(data):
    body

template popLoop(rx, data: typed, body: untyped): untyped =
  while not rx.tryPop(data):
    body

proc str(ch: char): String {.inline.} =
  result = initStringOfCap(1)
  result.add ch

proc pongFn {.thread.} =
  while true:
    var n: String
    popLoop(q1, n): cpuRelax()
    var p = isolate(n)
    pushLoop(q2, p): cpuRelax()
    #sleep 20
    if n[0] == '0': break
    #assert n == toStr("1")

proc pingPong =
  q1 = newSpscQueue[String](50)
  q2 = newSpscQueue[String](50)
  createThread(pong, pongFn)
  for i in 1..numIters:
    var p = isolate(str('1'))
    pushLoop(q1, p): cpuRelax()
    var n: String
    #sleep 10
    popLoop(q2, n): cpuRelax()
    #assert n == toStr("1")
  var p = isolate(str('0'))
  pushLoop(q1, p): cpuRelax()
  var n: String
  popLoop(q2, n): cpuRelax()
  #assert n == toStr("0")
  pong.joinThread()

pingPong()
