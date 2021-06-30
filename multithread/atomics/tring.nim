# --threads:on --panics:on --gc:arc -d:useMalloc -t:"-O3 -fsanitize=thread" -l:"-fsanitize=thread" -d:danger -g
import ring, std/isolation

const
  seed = 99
  bufCap = 20
  numIters = 1000

type
  Foo = ref object
    id: string

type
  ThreadArgs = object
    id: WorkerKind
    queue: ptr SpscQueue[Foo]

  WorkerKind = enum
    Producer
    Consumer

template work(expectedId: typed, body: untyped): untyped {.dirty.} =
  if args.id == expectedId:
    body

template pushLoop(queue, data: typed, body: untyped): untyped =
  while not queue.tryPush(data):
    body

template popLoop(queue, data: typed, body: untyped): untyped =
  while not queue.tryPop(data):
    body

proc threadFn(args: ThreadArgs) =
  work(Consumer):
    for i in 0 ..< numIters:
      var res: Foo
      args.queue[].popLoop(res): cpuRelax()
      #echo " >> popped ", res.id
      assert res.id == $(seed + i)
  work(Producer):
    for i in 0 ..< numIters:
      var p = isolate(Foo(id: $(i + seed)))
      args.queue[].pushLoop(p): cpuRelax()
      #echo " >> pushed ", $(i + seed)

proc testSpScRing =
  var
    queue: SpscQueue[Foo]
    thr1, thr2: Thread[ThreadArgs]
  init(queue, bufCap)
  createThread(thr1, threadFn, ThreadArgs(id: Producer, queue: addr queue))
  createThread(thr2, threadFn, ThreadArgs(id: Consumer, queue: addr queue))
  joinThread(thr1)
  joinThread(thr2)

testSpScRing()
