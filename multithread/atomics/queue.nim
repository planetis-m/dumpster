import std/isolation, sync/atomics2

type
  MpscQueue*[T] = object
    head: Atomic[ptr Node[T]]

  Node[T] = object
    value: T
    next: ptr Node[T]

proc push*[T](this: var MpscQueue[T]; value: sink Isolated[T]) {.nodestroy.} =
  var n = createSharedU(Node[T])
  n.value = extract value
  var staleHead = this.head.load(Relaxed)
  while true:
    n.next = staleHead
    if this.head.compareExchangeWeak(staleHead, n, Release):
      break

template push*[T](this: var MpscQueue[T]; value: T) =
  push(this, isolate(value))

proc popAllReverse[T](this: var MpscQueue[T]): ptr Node[T] {.inline.} =
  result = this.head.exchange(nil, Consume)

proc popAll[T](this: var MpscQueue[T]): ptr Node[T] =
  var
    last = popAllReverse(this)
  while last != nil:
    let tmp = last
    last = last.next
    tmp.next = result
    result = tmp

iterator items*[T](this: var MpscQueue[T]): T =
  var x = this.popAll()
  while x != nil:
    let tmp = x
    x = x.next
    yield tmp.value
    deallocShared(tmp)

when isMainModule:
  var q: MpscQueue[int]
  # insert elements
  q.push(42)
  q.push(2)
  # pop elements
  for x in q.items:
    echo " >> popped ", x
