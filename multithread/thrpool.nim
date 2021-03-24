# https://www.youtube.com/watch?v=7i9z4CRYLAE
import std / [locks, os, random, strformat]

const
  numThreads = 4

type
  Task = object
    taskFunc: proc (a, b: int) {.thread.}
    a, b: int

var
  threads: array[numThreads, Thread[void]]
  mutexQueue: Lock
  condQueue: Cond

  taskQueue: array[256, Task]
  taskCount = 0

proc sum(a, b: int) =
  sleep(1000)
  let sum = a + b
  echo &"Sum of {a} and {b} is {sum}"

proc product(a, b: int) =
  sleep(500)
  let prod = a * b
  echo &"Product of {a} and {b} is {prod}"

proc executeTask(task: Task) =
  task.taskFunc(task.a, task.b)

proc submitTask(task: Task) =
  acquire(mutexQueue)
  taskQueue[taskCount] = task
  taskCount.inc
  #echo "new Task addded"
  release(mutexQueue)
  signal(condQueue)

proc startThread =
  while true:
    acquire(mutexQueue)
    while taskCount == 0:
      wait(condQueue, mutexQueue)
    let task = taskQueue[0]
    for i in 0 ..< taskCount - 1:
      taskQueue[i] = taskQueue[i + 1]
    taskCount.dec
    release(mutexQueue)

    executeTask(task)

proc main =
  #randomize()
  initLock mutexQueue
  initCond condQueue

  for i in 0 ..< numThreads:
    createThread(threads[i], startThread)

  for i in 0 ..< 100:
    let t = Task(
      taskFunc: if i mod 2 == 0: sum else: product,
      a: rand(100),
      b: rand(100)
    )
    submitTask t

  joinThreads(threads)

  deinitCond(condQueue)
  deinitLock(mutexQueue)

main()
