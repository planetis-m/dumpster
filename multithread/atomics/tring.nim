# --threads:on --panics:on --gc:arc -d:useMalloc -t:"-O3 -fsanitize=thread" -l:"-fsanitize=thread" -d:danger -g
import ring

const
  seed = 99
  bufCap = 20
  numIters = 1000

var
  rng: SpscQueue[int]
  thr1, thr2: Thread[void]

proc producer =
  for i in 0 ..< numIters:
    while not rng.push(i + seed): cpuRelax()
    #echo " >> pushed ", i+seed

proc consumer =
  for i in 0 ..< numIters:
    var res: int
    while not rng.pop(res): cpuRelax()
    #echo " >> popped ", res
    assert res == seed + i

proc testSpScRing =
  init(rng, bufCap)
  createThread(thr1, producer)
  createThread(thr2, consumer)
  joinThread(thr1)
  joinThread(thr2)

testSpScRing()
