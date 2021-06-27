# --threads:on --panics:on --gc:arc -d:useMalloc -t:"-O3 -fsanitize=thread" -l:"-fsanitize=thread" -d:danger -g
import ring, std/isolation

const
  seed = 99
  bufCap = 20
  numIters = 1000

type
  Person = ref object
    name: string

var
  rng: SpscQueue[Person]
  thr1, thr2: Thread[void]

proc producer =
  const names = ["Dimitris", "Antonis", "Maria", "George"]
  for i in 0 ..< numIters:
    var p = Person(name: names[i and 3])
    echo " >> pushing ", p.name
    while not rng.push(p): cpuRelax()

proc consumer =
  for i in 0 ..< numIters:
    var res: Person
    while not rng.pop(res): cpuRelax()
    echo " >> popped ", res.name

proc testSpScRing =
  init(rng, bufCap)
  createThread(thr1, producer)
  createThread(thr2, consumer)
  joinThread(thr1)
  joinThread(thr2)

testSpScRing()
