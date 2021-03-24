# Port from https://github.com/dhylands/TimerUART/blob/master/CBUF.h
import math

type
  Buffer*[N: static[int]; T] = object
    head, tail: int
    data: array[N, T]

proc initBuffer*[N, T](): Buffer[N, T] =
  assert isPowerOfTwo(N)

proc len*[N, T](b: Buffer[N, T]): int {.inline.} =
  ## Return the number of elements of `b`.
  b.tail - b.head

proc isEmpty*[N, T](b: Buffer[N, T]): bool {.inline.} =
  ## Is the buffer empty.
  len(b) == 0

proc isFull*[N, T](b: Buffer[N, T]): bool {.inline.} =
  ## Is the buffer at capacity (add will overwrite another element)
  len(b) == N

proc hasError*[N, T](b: Buffer[N, T]): bool {.inline.} =
  ## Is the buffer overflowed or underflowed.
  len(b) > N

proc add*[N, T](b: var Buffer[N, T]; item: T) =
  ## Add an `item` to the end of the buffer.
  b.data[b.tail and (N - 1)] = item
  inc(b.tail)

proc pop*[N, T](b: var Buffer[N, T]): T =
  ## Remove and returns the first element of the buffer.
  result = b.data[b.head and (N - 1)]
  inc(b.head)

proc peekFirst*[N, T](b: Buffer[N, T]): T =
  ## Returns the first element of `b`, but does not remove it from the buffer.
  b.data[b.head and (N - 1)]

proc peekLast*[N, T](b: Buffer[N, T]): T =
  ## Returns the last element of `b`, but does not remove it from the buffer.
  b.data[(b.tail - 1) and (N - 1)]

proc `[]`*[N, T](b: Buffer[N, T]; i: Natural): T {.inline.} =
  ## Access the i-th element of `b` by order from first to last.
  ## b[0] is the first, b[^1] is the last.
  b.data[(b.head + i) and (N - 1)]

proc `[]`*[N, T](b: Buffer[N, T]; i: BackwardsIndex): T {.inline.} =
  b.data[(b.tail - int(i)) and (N - 1)]

iterator items*[N, T](b: Buffer[N, T]): T =
  ## Yield every element of `b`.
  var i = b.head and (N - 1)
  let len = len(b)
  for c in 0 ..< len:
    yield b.data[i]
    i = (i + 1) and (N - 1)
    assert len(b) == len, "buffer modified while iterating over it"

iterator pairs*[N, T](b: Buffer[N, T]): tuple[key: int; val: T] =
  ## Yield every (position, value) of `b`.
  var i = b.head and (N - 1)
  let len = len(b)
  for c in 0 ..< len:
    yield (c, b.data[i])
    i = (i + 1) and (N - 1)
    assert len(b) == len, "buffer modified while iterating over it"

when isMainModule:
  var b = initBuffer[4, int]()
  assert b.isEmpty
  var s: seq[int] = @[]
  for i in 1 .. 4:
    b.add(i)
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
    b.add(i)
  s = @[]
  for x in b:
    s.add(x)
  assert s == @[4, 5, 6]
  # From here on operations will fail
  for i in 5 .. 7:
    b.add(i)
  assert b.hasError

  b = initBuffer[4, int]()
  b.add(2)
  b.add(3)
  for x in b:
    echo x
    discard b.pop
  for x in b:
    assert false
