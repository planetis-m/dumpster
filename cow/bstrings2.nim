import sync/spsc_queue

const
  numIters = 1000000

var
  pong: Thread[void]
  q1: SpscQueue[string]
  q2: SpscQueue[string]

template pushLoop(tx, data: typed, body: untyped): untyped =
  while not tx.tryPush(data):
    body

template popLoop(rx, data: typed, body: untyped): untyped =
  while not rx.tryPop(data):
    body

proc pongFn {.thread.} =
  while true:
    var n: string
    popLoop(q1, n): cpuRelax()
    pushLoop(q2, n): cpuRelax()
    #sleep 20
    if n == "0": break
    assert n == "1"

proc pingPong =
  q1 = newSpscQueue[string](50)
  q2 = newSpscQueue[string](50)
  createThread(pong, pongFn)
  for i in 1..numIters:
    var m = "1"
    prepareMutation m
    pushLoop(q1, m): cpuRelax()
    var n: string
    #sleep 10
    popLoop(q2, n): cpuRelax()
    assert n == "1"
  var m = "0"
  prepareMutation m
  pushLoop(q1, m): cpuRelax()
  var n: string
  popLoop(q2, n): cpuRelax()
  assert n == "0"
  pong.joinThread()

pingPong()
