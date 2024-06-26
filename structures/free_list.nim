# https://www.gingerbill.org/article/2021/11/30/memory-allocation-strategies-005/
# https://youtu.be/UTii4dyhR5c
type
  AllocationHeader = object
    blockSize: int
    padding: uint8

  FreeNodeObj = object
    blockSize: int
    next: FreeNode
  FreeNode = ptr FreeNodeObj

  PlacementPolicy = enum
    FindFirst
    FindBest

  FreeList = object
    policy: PlacementPolicy
    used: int
    head: FreeNode
    bufLen: int
    buf: ptr UncheckedArray[byte]

const
  DefaultAlignment = 8
  MaxAlignment = 128

proc freeAll*(x: var FreeList) =
  x.used = 0
  let firstNode = cast[FreeNode](x.buf)
  firstNode.blockSize = x.bufLen
  firstNode.next = nil
  x.head = firstNode

proc init*(x: var FreeList, buffer: openarray[byte]; policy = FindFirst) =
  x.policy = policy
  x.buf = cast[ptr UncheckedArray[byte]](buffer)
  x.bufLen = buffer.len
  freeAll(x)

proc nodeInsert(head: var FreeNode, prevNode, newNode: FreeNode) =
  if prevNode == nil:
    newNode.next = head
    head = newNode
  else:
    newNode.next = prevNode.next
    prevNode.next = newNode

proc nodeRemove(head: var FreeNode, prevNode, delNode: FreeNode) =
  if prevNode == nil:
    head = delNode.next
  else:
    prevNode.next = delNode.next

proc coalescence(x: var FreeList, prevNode, freeNode: FreeNode) =
  if freeNode.next != nil and
      (cast[uint](freeNode) + freeNode.blockSize.uint == cast[uint](freeNode.next)):
    inc freeNode.blockSize, freeNode.next.blockSize
    nodeRemove(x.head, freeNode, freeNode.next)
  if prevNode != nil and
      (cast[uint](prevNode) + prevNode.blockSize.uint == cast[uint](freeNode)):
    inc prevNode.blockSize, freeNode.blockSize
    nodeRemove(x.head, prevNode, freeNode)

proc alignup(n, align: uint): uint {.inline.} =
  (n + align - 1) and (not (align - 1))

proc calcPaddingWithHeader(address, align: uint, headerSize: int): int =
  # assert isPowerOfTwo(align.int)
  let
    neededSpace = headerSize.uint
    headerAddress = address + neededSpace
    alignedAddress = alignup(headerAddress, align)
    padding = alignedAddress - address
  result = padding.int

proc findFirst(x: FreeList, size, align: int): tuple[node, prev: FreeNode, padding: int] =
  var
    node = x.head
    prevNode: FreeNode = nil
    padding = 0
  while node != nil:
    padding = calcPaddingWithHeader(cast[uint](node), align.uint, sizeof(AllocationHeader))
    let requiredSpace = size + padding
    if node.blockSize >= requiredSpace:
      break
    prevNode = node
    node = node.next
  result = (node, prevNode, padding)

proc findBest(x: FreeList, size, align: int): tuple[node, prev: FreeNode, padding: int] =
  var
    smallestDiff = high(int)
    node = x.head
    prevNode: FreeNode = nil
    bestNode: FreeNode = nil
    padding = 0
  while node != nil:
    padding = calcPaddingWithHeader(cast[uint](node), align.uint, sizeof(AllocationHeader))
    let requiredSpace = size + padding
    if node.blockSize >= requiredSpace and (node.blockSize - requiredSpace < smallestDiff):
      bestNode = node
      smallestDiff = node.blockSize - requiredSpace
    prevNode = node
    node = node.next
  result = (bestNode, prevNode, padding)

proc alignedAlloc*(x: var FreeList, size, align: Natural): pointer =
  var
    padding = 0
    prevNode: FreeNode = nil
    node: FreeNode = nil
    alignmentPadding, requiredSpace, remaining: int
    headerPtr: ptr AllocationHeader
  let adjustedSize = max(size, sizeof(FreeNodeObj))
  let adjustedAlign = clamp(align, DefaultAlignment, MaxAlignment)
  if x.policy == FindBest:
    (node, prevNode, padding) = findBest(x, adjustedSize, adjustedAlign)
  else:
    (node, prevNode, padding) = findFirst(x, adjustedSize, adjustedAlign)
  if node == nil:
    assert false, "Free list has no free memory"
    return nil
  alignmentPadding = padding - sizeof(AllocationHeader)
  requiredSpace = adjustedSize + padding
  remaining = node.blockSize - requiredSpace
  if remaining >= sizeof(FreeNodeObj):
    let newNode = cast[FreeNode](cast[uint](node) + requiredSpace.uint)
    newNode.blockSize = remaining
    nodeInsert(x.head, node, newNode)
  else:
    inc requiredSpace, remaining
  nodeRemove(x.head, prevNode, node)
  headerPtr = cast[ptr AllocationHeader](cast[uint](node) + alignmentPadding.uint)
  headerPtr.blockSize = requiredSpace
  headerPtr.padding = alignmentPadding.uint8
  inc x.used, requiredSpace
  result = cast[pointer](cast[uint](headerPtr) + sizeof(AllocationHeader).uint)

proc alloc*(x: var FreeList; size: Natural): pointer =
  alignedAlloc(x, size, DefaultAlignment)

proc free*(x: var FreeList, p: pointer) =
  if p == nil:
    return
  let header = cast[ptr AllocationHeader](cast[uint](p) - sizeof(AllocationHeader).uint)
  let padding = header.padding
  var freeNode = cast[FreeNode](cast[uint](header) - padding.uint)
  freeNode.blockSize = header.blockSize
  freeNode.next = nil
  var
    node = x.head
    prevNode: FreeNode = nil
  while node != nil:
    if cast[uint](node) > cast[uint](p):
      nodeInsert(x.head, prevNode, freeNode)
      break
    prevNode = node
    node = node.next
  if node == nil:
    nodeInsert(x.head, prevNode, freeNode)
  dec x.used, freeNode.blockSize
  coalescence(x, prevNode, freeNode)

when isMainModule:
  const BufferSize = 1024
  var backingBuffer: array[BufferSize, byte]
  var x: FreeList
  init(x, backingBuffer)

  assert x.used == 0
  assert x.head != nil
  assert x.head.blockSize == BufferSize
  assert x.head.next == nil

  let p1 = x.alloc(100)
  assert p1 != nil
  assert x.used > 100

  let p2 = x.alloc(200)
  assert p2 != nil
  assert cast[uint](p2) > cast[uint](p1)

  x.free(p1)
  assert x.head != nil

  x.free(p2)
  assert x.used == 0
  assert x.head.blockSize == BufferSize
