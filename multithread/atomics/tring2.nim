import ring, std/isolation, os

const
  seed = 99
  bufCap = 20

type
  Foo = ref FooObj
  FooObj = object
    id: string

var
  count = 0

proc `=destroy`(this: var FooObj) =
  inc count
  echo count, " ", this.id
  `=destroy`(this.id)

var
  rng: SpscQueue[Foo]
  thr1, thr2: Thread[void]

proc producer =
  for i in 0 ..< bufCap+3:
    var p = isolate(Foo(id: $(i + seed)))
    while not rng.tryPush(p): cpuRelax()
    #echo " >> pushed ", $(i + seed)

proc consumer =
  for i in 0 ..< 3:
    var res: Foo
    while not rng.tryPop(res): cpuRelax()
    echo " >> popped ", res.id
    sleep 2

proc testSpScRing =
  init(rng, bufCap)
  createThread(thr1, producer)
  createThread(thr2, consumer)
  joinThread(thr1)
  joinThread(thr2)

testSpScRing()
