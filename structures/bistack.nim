type
  BiStack* = object # Double-ended stack (aka Deque)
    front, back: int
    bufLen: int
    buf: ptr UncheckedArray[byte]

const
  DefaultAlignment = 8

proc alignup(n, align: uint): uint {.inline.} =
  (n + align - 1) and (not (align - 1))

proc init*(s: var BiStack; buffer: openarray[byte]) =
  s.buf = cast[ptr UncheckedArray[byte]](buffer)
  s.bufLen = buffer.len
  s.front = 0
  s.back = buffer.len

proc alignedAllocFront*(s: var BiStack, size, align: Natural): pointer =
  let
    currAddr = cast[uint](s.buf) + s.front.uint
    alignedAddr = alignup(currAddr, align.uint)
    padding = int(alignedAddr - currAddr)
  if s.front + padding + size > s.back:
    # Stack allocator is out of memory
    return nil
  s.front = s.front + size + padding
  result = cast[pointer](alignedAddr)
  zeroMem(result, size)

proc allocFront*(s: var BiStack; size: Natural): pointer =
  alignedAllocFront(s, size, DefaultAlignment)

proc alignedAllocBack*(s: var BiStack, size, align: Natural): pointer =
  let
    currAddr = cast[uint](s.buf) + s.front.uint
    alignedAddr = alignup(currAddr, align.uint)
    padding = (alignedAddr - currAddr).int
  if s.back - padding - size < s.front:
    # Stack allocator is out of memory
    return nil
  s.back = s.back - size - padding
  result = cast[pointer](alignedAddr)
  zeroMem(result, size)

proc allocBack*(s: var BiStack; size: Natural): pointer =
  alignedAllocBack(s, size, DefaultAlignment)

proc resetFront*(s: var BiStack) =
  s.front = 0

proc resetBack*(s: var BiStack) =
  s.back = s.bufLen

proc resetAll*(s: var BiStack) =
  resetBack(s)
  resetFront(s)

proc margins*(s: BiStack): int {.inline.} =
  result = s.back - s.front

when isMainModule:
  var backingBuffer {.align: DefaultAlignment.}: array[16, byte]
  var s: BiStack
  init(s, backingBuffer)

  # Allocate all available memory from the front
  assert s.allocFront(16) != nil
  # Try to allocate more memory from the front
  assert s.allocFront(1) == nil
  # Try to allocate memory from the back
  assert s.allocBack(1) == nil
  # Reset and allocate all available memory from the back
  s.resetAll()
  assert s.allocBack(14) != nil
  # Try to allocate memory from the front
  assert s.allocFront(4) == nil
  assert s.allocFront(2) != nil
  # Try to allocate more memory from the back
  assert s.allocBack(1) == nil
  # Check that the margin is indeed 0
  assert s.margins() == 0
