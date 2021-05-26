# Port from https://github.com/dhylands/TimerUART/blob/master/CBUF.h
# todo: make an unsigned generic index type
import math

type
  Buffer*[Cap: static[int]; T] = object
    head, tail: int
    data: array[Cap, T]

proc initBuffer*[Cap, T](): Buffer[Cap, T] =
  assert isPowerOfTwo(Cap) #and Cap <= 1 shl (sizeof(head) * 8 - 1)

proc len*[Cap, T](b: Buffer[Cap, T]): int {.inline.} =
  ## Return the number of elements of `b`.
  b.head - b.tail

proc isEmpty*[Cap, T](b: Buffer[Cap, T]): bool {.inline.} =
  ## Is the buffer empty.
  len(b) == 0

proc isFull*[Cap, T](b: Buffer[Cap, T]): bool {.inline.} =
  ## Is the buffer at capacity (push will overwrite another element)
  len(b) == Cap

template mask(val: untyped): untyped = val and (Cap - 1)

proc push*[Cap, T](b: var Buffer[Cap, T]; item: T) =
  ## Add an `item` to the end of the buffer.
  assert not isFull(b)
  b.data[mask b.head] = item
  inc b.head

proc pop*[Cap, T](b: var Buffer[Cap, T]): T =
  ## Remove and returns the first element of the buffer.
  assert not isEmpty(b)
  result = b.data[mask b.tail]
  inc b.tail

proc peekFirst*[Cap, T](b: Buffer[Cap, T]): T =
  ## Returns the first element of `b`, but does not remove it from the buffer.
  b.data[mask b.tail]

proc peekLast*[Cap, T](b: Buffer[Cap, T]): T =
  ## Returns the last element of `b`, but does not remove it from the buffer.
  b.data[mask(b.head - 1)]

proc `[]`*[Cap, T](b: Buffer[Cap, T]; i: Natural): T {.inline.} =
  ## Access the i-th element of `b` by order from first to last.
  ## b[0] is the first, b[^1] is the last.
  b.data[mask(b.tail + i)]

proc `[]`*[Cap, T](b: Buffer[Cap, T]; i: BackwardsIndex): T {.inline.} =
  b.data[mask(b.head - int(i))]

iterator items*[Cap, T](b: Buffer[Cap, T]): T =
  ## Yield every element of `b`.
  var i = b.tail
  let len = len(b)
  for c in 0 ..< len:
    yield b.data[mask i]
    inc i
    assert len(b) == len, "buffer modified while iterating over it"

iterator pairs*[Cap, T](b: Buffer[Cap, T]): tuple[key: int; val: T] =
  ## Yield every (position, value) of `b`.
  var i = b.tail
  let len = len(b)
  for c in 0 ..< len:
    yield (c, b.data[mask i])
    inc i
    assert len(b) == len, "buffer modified while iterating over it"

when isMainModule:
  var b = initBuffer[4, int]()
  assert b.isEmpty
  var s: seq[int] = @[]
  for i in 0 ..< b.Cap:
    b.push(i + 1)
  for x in b:
    s.add(x)
  assert b.isFull
  assert s == @[1, 2, 3, 4]
  assert b.peekLast == s[^1]
  assert b.peekFirst == s[0]
  for i in 0 .. 3:
    assert b[i] == s[i]
  for i in 1 .. 4:
    assert b[^i] == s[^i]
  for x in s:
    assert b.pop == x
  assert b.isEmpty
  for i in 4 .. 6:
    b.push(i)
  s = @[]
  for x in b:
    s.add(x)
  assert s == @[4, 5, 6]
