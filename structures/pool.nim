# https://www.gingerbill.org/code/memory-allocation-strategies/part004.c

import sanitizers

type
  FreeNode = object
    next: ptr FreeNode

  FixedPool*[T] = object
    chunkSize: int
    head: ptr FreeNode # Free List Head
    bufLen: int
    buf: ptr UncheckedArray[byte]

const
  DefaultAlignment = 8

template guardedAccess(p: ptr FreeNode, body: untyped): untyped =
  unpoisonMemRegion(p, sizeof(FreeNode))
  body
  poisonMemRegion(p, sizeof(FreeNode))

proc alignup(n, align: uint): uint {.inline.} =
  (n + align - 1) and (not (align - 1))

proc deallocAll*(x: var FixedPool)

proc init*[T](x: var FixedPool[T], buffer: openarray[byte]) =
  let start = cast[uint](buffer)
  let maxAlign = max(alignof(T), alignof(FreeNode)).uint
  let alignedStart = alignup(start, maxAlign)
  let alignedLen = buffer.len - int(alignedStart - start)
  # Align chunk size up to the required chunkAlignment
  let alignedSize = alignup(sizeof(T).uint, maxAlign).int
  # Assert that the parameters passed are valid
  assert alignedSize >= sizeof(FreeNode), "Chunk size is too small"
  assert alignedLen >= alignedSize, "Backing buffer length is smaller than the chunk size"
  # Store the adjusted parameters
  x.buf = cast[ptr UncheckedArray[byte]](alignedStart)
  x.bufLen = alignedLen
  x.chunkSize = alignedSize
  x.head = nil # Free List Head
  # Set up the free list for free chunks
  deallocAll(x)

proc alloc*[T](x: var FixedPool[T]): ptr T =
  # Get latest free node
  let node = x.head
  if node == nil:
    assert false, "FixedPool allocator has no free memory"
    return nil
  # Pop free node
  guardedAccess(node):
    x.head = node.next
  # Zero memory by default
  unpoisonMemRegion(node, sizeof(T))
  zeroMem(node, sizeof(T))
  result = cast[ptr T](node)

proc dealloc*[T](x: var FixedPool[T], p: ptr T) =
  if p == nil:
    # Ignore NULL pointers
    return
  let start = cast[uint](x.buf)
  let endAddr = start + uint(x.bufLen)
  if start > cast[uint](p) or cast[uint](p) >= endAddr:
    assert false, "Memory is out of bounds of the buffer in this pool"
    return
  poisonMemRegion(p, sizeof(T))
  # Push free node
  let node = cast[ptr FreeNode](p)
  guardedAccess(node):
    node.next = x.head
  x.head = node

proc deallocAll*(x: var FixedPool) =
  let chunkCount = x.bufLen div x.chunkSize
  # Set all chunks to be free
  for i in 0 ..< chunkCount:
    let p = cast[pointer](cast[uint](x.buf) + uint(i * x.chunkSize))
    let node = cast[ptr FreeNode](p)
    # Push free node onto the free list
    guardedAccess(node):
      node.next = x.head
    x.head = node
  poisonMemRegion(x.buf, x.bufLen)

when isMainModule:
  type
    Vector2D = object
      x, y: float32
      used: bool

  var backingBuffer {.align: DefaultAlignment.}: array[1024, byte]
  var x: FixedPool[Vector2D]
  init(x, backingBuffer)

  var a = alloc(x)
  let b = alloc(x)
  let c = alloc(x)
  var d = alloc(x)
  let e = alloc(x)
  let f = alloc(x)

  dealloc(x, f)
  dealloc(x, c)
  dealloc(x, b)
  dealloc(x, d)

  d = alloc(x)

  dealloc(x, a)

  a = alloc(x)

  dealloc(x, e)
  dealloc(x, a)
  dealloc(x, d)
