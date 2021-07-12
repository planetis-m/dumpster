from typetraits import supportsCopyMem

type
  Chunk[T] = object
    next: ptr Chunk[T]
    len: int
    elems: UncheckedArray[T]

  Pool*[T] = object ## A pool of 'T' nodes.
    len: int
    last: ptr Chunk[T]
    lastCap: int

  Node[T] = distinct ptr[T]

#proc `=copy`*[T](dest: var Node[T]; src: Node[T]) {.error.}

proc `=destroy`*[T](x: var Node[T]) =
  `=destroy`((ptr T)(x)[])

proc `=copy`*[T](dest: var Pool[T]; src: Pool[T]) {.error.}

proc `=destroy`*[T](p: var Pool[T]) =
  var it = p.last
  while it != nil:
    let next = it.next
    deallocShared(it)
    it = next

proc newNode*[T](p: var Pool[T]): Node[T] =
  if p.len >= p.lastCap:
    if p.lastCap == 0: p.lastCap = 4
    elif p.lastCap < 65_000: p.lastCap *= 2
    when not supportsCopyMem(T):
      let n = cast[ptr Chunk[T]](allocShared0(sizeof(Chunk[T]) + p.lastCap * sizeof(T)))
    else:
      let n = cast[ptr Chunk[T]](allocShared(sizeof(Chunk[T]) + p.lastCap * sizeof(T)))
    n.next = nil
    n.next = p.last
    p.last = n
    p.len = 0
  result = Node[T](addr(p.last.elems[p.len]))
  inc p.len
  inc p.last.len

type
  Foo = object
    s: string

proc `=destroy`(x: var Foo) =
  echo "destroying ", x.s
  `=destroy`(x.s)

proc test(p: var Pool[Foo]) =
  let x = newNode(p)
  (ptr Foo)(x).s = "Hello World"

proc main =
  var p: Pool[Foo]
  test(p)
  echo "exit"

main()
