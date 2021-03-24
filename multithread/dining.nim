import locks, random, os

const
  philosophers = 5

type
  State = enum
    Thinking, Eating

template left: untyped = (i - 1) mod philosophers
template right: untyped = (i + 1) mod philosophers

var
  exit = false
  p: array[philosophers, Thread[int]]
  infot: Thread[void]
  state: array[philosophers, State] # array keeping everyone's state
  mutex: Lock
  pickUpCond: Cond # queue for a philosopher thread to enter if it fails to pickup both chopsticks

proc pickup(i: int) =
  var tmp = left
  if tmp < 0:
    tmp += philosophers
  acquire(mutex)
  while state[tmp] == Eating or
      state[right] == Eating:
    wait(pickUpCond, mutex)
  # none of the adjacent neighbours is eating
  state[i] = Eating
  release(mutex)

proc putdown(i: int) =
  acquire(mutex)
  state[i] = Thinking
  signal(pickUpCond)
  release(mutex)

proc think(i: int) =
  sleep(rand(3000))
proc eat(i: int) =
  sleep(rand(2000))

proc philosopher(i: int) =
  echo "Philosopher :", i
  while not exit:
    think(i)
    pickup(i)
    eat(i)
    putdown(i)

proc printInfo =
  var
    n = 0
    a: array[2, int] # no two adjacent philosophers eat at the same time
  acquire(mutex)
  for i in 0 ..< philosophers:
    if state[i] == Eating:
      a[n] = i
      n.inc
  release(mutex)
  var buf = ""
  for k in 0 ..< n:
    buf.addInt a[k]
    if k < n - 1:
      buf.add ", "
  echo "# of philosophers eating is ", n, ": ", buf

proc info =
  while not exit:
    sleep(1000)
    printInfo()

proc main =
  #randomize()
  initLock mutex
  initCond pickUpCond

  for i in 0 ..< philosophers:
    createThread(p[i], philosopher, i)
  createThread(infot, info)

  joinThreads(p)
  joinThread(infot)

  deinitCond pickUpCond
  deinitLock mutex

setControlCHook(proc () {.noconv.} =
  exit = true
  echo "Quiting, please wait...."
)

main()
