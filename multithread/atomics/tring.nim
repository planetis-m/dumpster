# --threads:on --panics:on --gc:arc -d:useMalloc -t:"-O3 -fsanitize=thread" -l:"-fsanitize=thread" -d:danger -g
import ring, std/isolation

const
  seed = 99
  bufCap = 20
  numIters = 1000

type
  Foo = ref object
    id: string

var
  rng: SpscQueue[Foo]
  thr1, thr2: Thread[void]

proc producer =
  for i in 0 ..< numIters:
    var p = isolate(Foo(id: $(i + seed))) # crashes
    while not rng.tryPush(p): cpuRelax()
    #while not rng.tryPush(Foo(id: $(i + seed))): cpuRelax()
    #echo " >> pushed ", $(i + seed)

proc consumer =
  for i in 0 ..< numIters:
    var res: Foo
    while not rng.tryPop(res): cpuRelax()
    #echo " >> popped ", res.id
    assert res.id == $(seed + i)

proc testSpScRing =
  init(rng, bufCap)
  createThread(thr1, producer)
  createThread(thr2, consumer)
  joinThread(thr1)
  joinThread(thr2)

testSpScRing()
