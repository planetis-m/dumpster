from typetraits import supportsCopyMem
import expsmarts, std/isolation, threading/atomics

type
  Chunk[T] = object
    next: ptr Chunk[T]
    len: int
    elems: UncheckedArray[Payload[T]]

  Pool*[T] = object ## A pool of 'T' nodes.
    len: int
    last: ptr Chunk[T]
    lastCap: int

proc `=destroy`*[T](p: var Pool[T]) =
  var it = p.last
  while it != nil:
    let next = it.next
    deallocShared(it)
    it = next

proc `=copy`*[T](dest: var Pool[T]; src: Pool[T]) {.error.}

proc newSharedPtr*[T](p: var Pool[T]; val: sink Isolated[T]): SharedPtr[T] {.nodestroy.} =
  if p.len >= p.lastCap:
    if p.lastCap == 0: p.lastCap = 4
    elif p.lastCap < 65_000: p.lastCap *= 2
    let n = cast[ptr Chunk[T]](allocShared(sizeof(Chunk[T]) + p.lastCap * sizeof(Payload[T])))
    n.next = nil
    n.next = p.last
    p.last = n
    p.len = 0
  result.val = addr(p.last.elems[p.len])
  int(result.val.counter) = 0
  result.val.value = extract val
  inc p.len
  inc p.last.len

when isMainModule:
  type
    Foo = object
      s: string

  proc `=destroy`(x: var Foo) =
    echo "destroying ", x.s
    `=destroy`(x.s)

  proc test(p: var Pool[Foo]) =
    let x = newSharedPtr(p, isolate(Foo(s: "Hello World")))

  proc main =
    var p: Pool[Foo]
    test(p)
    echo "exit"

  main()
