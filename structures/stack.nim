# https://www.gingerbill.org/article/2019/02/15/memory-allocation-strategies-003/
type
  Stack* = object
    buf: ptr UncheckedArray[byte]
    bufLen, offset: int

  StackMarker* = distinct int

const
  DefaultAlignment = 8

proc alignup(n, align: uint): uint {.inline.} =
  (n + align - 1) and (not (align - 1))

proc init*(s: var Stack; buffer: openarray[byte]) =
  s.buf = cast[ptr UncheckedArray[byte]](buffer)
  s.bufLen = buffer.len
  s.offset = 0

proc alignedAlloc*(s: var Stack; size, align: Natural): pointer =
  let
    currAddr = cast[uint](s.buf) + s.offset.uint
    alignedAddr = alignup(currAddr, align.uint)
    padding = (alignedAddr - currAddr).int
  if s.offset + padding + size > s.bufLen:
    # Stack allocator is out of memory
    return nil
  s.offset = s.offset + size + padding
  result = cast[pointer](alignedAddr)
  zeroMem(result, size)

proc alloc*(s: var Stack; size: Natural): pointer =
  alignedAlloc(s, size, DefaultAlignment)

proc freeToMarker*(s: var Stack; marker: StackMarker) =
  assert(marker.int >= 0 and marker.int <= s.offset)
  s.offset = marker.int

proc getMarker*(s: Stack): StackMarker =
  result = StackMarker(s.offset)

proc freeAll*(s: var Stack) =
  s.offset = 0

when isMainModule:
  type
    Vector2D = object
      x, y: float32

    LargeStruct = object
      data: array[100, int32]

    SmallStruct = object
      value: int8

  var backingBuffer: array[1024, byte]
  var s: Stack
  init(s, backingBuffer)

  # Allocate different sizes
  let a = cast[ptr Vector2D](alloc(s, sizeof(Vector2D)))
  let b = cast[ptr LargeStruct](alloc(s, sizeof(LargeStruct)))
  let c = cast[ptr SmallStruct](alloc(s, sizeof(SmallStruct)))
  let d = cast[ptr int](alloc(s, sizeof(int)))
  let e = cast[ptr float64](alloc(s, sizeof(float64)))

  # Set some values
  a.x = 1.0
  a.y = 2.0
  b.data[0] = 42
  c.value = 7
  d[] = 100
  e[] = 3.14

  # Print current offset
  echo "Current offset after allocations: ", s.offset

  # Get a marker
  let marker = getMarker(s)

  # Allocate more
  let f = cast[ptr Vector2D](alloc(s, sizeof(Vector2D)))
  f.x = 5.0
  f.y = 6.0

  echo "Offset after allocating f: ", s.offset

  # Free back to marker
  freeToMarker(s, marker)

  echo "Offset after freeing to marker: ", s.offset

  # Allocate again
  let g = cast[ptr SmallStruct](alloc(s, sizeof(SmallStruct)))
  g.value = 9

  echo "Final offset: ", s.offset

  # Print values to ensure they're still correct
  assert a.x == 1.0 and a.y == 2.0
  assert b.data[0] == 42
  assert c.value == 7
  assert d[] == 100
  assert e[] == 3.14
  assert g.value == 9

  freeAll(s)
  assert s.offset == 0
