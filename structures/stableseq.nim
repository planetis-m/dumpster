type
  StableChunk[T] = object
    next: ptr StableChunk[T]
    len, cap: int
    data: UncheckedArray[T]

  StableSeq*[T] = object
    head, tail: ptr StableChunk[T]
    current: ptr StableChunk[T]
    currentIndex: int
    len: int

proc createChunk[T](cap: int): ptr StableChunk[T] =
  result = cast[ptr StableChunk[T]](alloc0(sizeof(StableChunk[T]) + sizeof(T) * cap))
  result.cap = cap

proc createStableSeq*[T](cap = 10): StableSeq[T] =
  let c = createChunk[T](cap)
  StableSeq[T](head: c, tail: c, current: c, currentIndex: 0, len: 0)

proc destroy*[T](s: StableSeq[T]) =
  var it = s.head
  while it != nil:
    let next = it.next
    dealloc it
    it = next

proc append*[T](s: var StableSeq[T]; data: sink T): ptr T =
  if s.tail.len >= s.tail.cap:
    let oldTail = s.tail
    let newCap = oldTail.cap * 3 div 2
    assert newCap > 2
    s.tail = createChunk[T](newCap)
    oldTail.next = s.tail

  result = addr s.tail.data[s.tail.len]
  result[] = ensureMove data
  inc s.tail.len
  inc s.len

proc add*[T](s: var StableSeq[T]; data: sink T) =
  discard append(s, data)

iterator items*[T](s: StableSeq[T]): lent T =
  var it = s.head
  while it != nil:
    for i in 0 ..< it.len:
      yield it.data[i]
    it = it.next

proc nav[T](s: var StableSeq[T]; i: int): ptr StableChunk[T] =
  if i < s.currentIndex:
    s.current = s.head
    s.currentIndex = 0

  while s.current != nil:
    if i < s.currentIndex + s.current.len:
      return s.current
    inc s.currentIndex, s.current.len
    s.current = s.current.next
  return nil

proc `[]`*[T](s: var StableSeq[T]; index: int): var T =
  let it = nav(s, index)
  if it != nil:
    return it.data[index - s.currentIndex]
  else:
    raise newException(IndexDefect, "index out of bounds")

proc `[]=`*[T](s: var StableSeq[T]; index: int; elem: sink T) =
  let it = nav(s, index)
  if it != nil:
    it.data[index - s.currentIndex] = elem
  else:
    raise newException(IndexDefect, "index out of bounds")

when isMainModule:
  var s = createStableSeq[int]()
  for i in 0 ..< 1000:
    discard s.append i

  # Test reading values
  echo "s[0] = ", s[0]
  echo "s[500] = ", s[500]
  echo "s[998] = ", s[998]

  # Test writing values
  s[0] = 5
  s[500] = 7
  s[998] = 13

  for elem in s:
    echo elem

  for i in countup(0, 990, 100):
    echo "s[", i, "] = ", s[i]

  destroy s
