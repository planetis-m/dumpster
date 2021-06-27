# --threads:on --panics:on --gc:arc -d:useMalloc -t:"-O3 -fsanitize=thread" -l:"-fsanitize=thread" -d:danger -g
import ring, std/isolation

const
  seed = 99
  bufCap = 20
  numIters = 1000

type
  Person = object
    id: string

var
  rng: SpscQueue[Person]
  thr1, thr2: Thread[void]

proc producer =
  for i in 0 ..< numIters:
    let p = Person(id: $(i + seed))
    while not rng.push(p): cpuRelax()
    #echo " >> pushed ", p.id

proc consumer =
  for i in 0 ..< numIters:
    var res: Person
    while not rng.pop(res): cpuRelax()
    #echo " >> popped ", res.id
    assert res.id == $(seed + i)

proc testSpScRing =
  init(rng, bufCap)
  createThread(thr1, producer)
  createThread(thr2, consumer)
  joinThread(thr1)
  joinThread(thr2)

testSpScRing()
