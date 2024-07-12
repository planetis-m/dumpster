from std/typetraits import supportsCopyMem
import strutils

type
  PoolElement[T] = object
    data: T
    used: bool

  FreeNode = object
    next: ptr FreeNode

  FixedPool*[T] = object
    chunkSize: int
    head: ptr FreeNode # Free List Head
    bufLen: int
    buf: ptr UncheckedArray[byte]

const
  DefaultAlignment = 8

proc alignUp(n: uint, align: int): uint {.inline.} =
  (n + align.uint - 1) and not (align.uint - 1)

proc deallocAll*[T](x: var FixedPool[T])

proc init*[T](x: var FixedPool[T], buffer: openarray[byte]) =
  let start = cast[uint](buffer)
  let maxAlign = max(alignof(T), alignof(FreeNode))
  let alignedStart = alignup(start, maxAlign)
  let alignedLen = buffer.len - int(alignedStart - start)
  # Align chunk size up to the required chunkAlignment
  when not supportsCopyMem(T):
    let alignedSize = alignup(sizeof(PoolElement[T]).uint, maxAlign).int
  else:
    let alignedSize = alignup(sizeof(T).uint, maxAlign).int
  # Assert that the parameters passed are valid
  assert sizeof(T) >= sizeof(FreeNode), "Chunk size is too small"
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
  x.head = node.next
  # Zero memory by default
  when not supportsCopyMem(T):
    let e = cast[ptr PoolElement[T]](node)
    assert not e.used # Forgot to free
    e.used = true
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
  # Push free node
  when not supportsCopyMem(T):
    let e = cast[ptr PoolElement[T]](p)
    assert e.used # Catch double-free
    `=destroy`(e.data)
    e.used = false
  let node = cast[ptr FreeNode](p)
  node.next = x.head
  x.head = node

proc deallocAll*[T](x: var FixedPool[T]) =
  let chunkCount = x.bufLen div x.chunkSize
  # Set all chunks to be free
  for i in 0 ..< chunkCount:
    let p = cast[pointer](cast[uint](x.buf) + uint(i * x.chunkSize))
    when not supportsCopyMem(T):
      let e = cast[ptr PoolElement[T]](p)
      if e.used: `=destroy`(e.data)
      e.used = false
    let node = cast[ptr FreeNode](p)
    # Push free node onto the free list
    node.next = x.head
    x.head = node

when isMainModule:
  type
    Vector2D = object
      x, y: float32
      z: bool

  proc `=destroy`(v: Vector2D) =
    echo "destroying object"

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
