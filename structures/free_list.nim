# https://www.gingerbill.org/article/2021/11/30/memory-allocation-strategies-005/
# https://youtu.be/UTii4dyhR5c
# Total sh*t
type
  AllocationHeader = object
    blockSize: int
    padding: int

  FreeNodeObj = object
    next: FreeNode
    blockSize: int
  FreeNode = ptr FreeNodeObj

  PlacementPolicy = enum
    FindFirst
    FindBest

  FreeList = object
    used: int
    head: FreeNode
    policy: PlacementPolicy
    bufLen: int
    buf: ptr UncheckedArray[byte]

const
  DefaultAlignment = 8

proc freeAll*(x: var FreeList) =
  x.used = 0
  let firstNode = cast[FreeNode](x.buf)
  firstNode.blockSize = x.bufLen
  firstNode.next = nil
  x.head = firstNode

proc init*(x: var FreeList, buffer: openarray[byte]) =
  x.buf = cast[ptr UncheckedArray[byte]](buffer)
  x.bufLen = buffer.len
  freeAll(x)

proc calcPaddingWithHeader(p, align: uint, headerSize: int): int
proc coalescence(fl: var FreeList, prevNode, freeNode: FreeNode)
proc nodeInsert(phead: var FreeNode, prevNode, newNode: FreeNode)
proc nodeRemove(phead: var FreeNode, prevNode, delNode: FreeNode)

proc findFirst(x: FreeList, size, align: int): tuple[node, prev: FreeNode, padding: int] =
  var
    node = x.head
    prev: FreeNode = nil
    padding = 0
  while node != nil:
    padding = calcPaddingWithHeader(cast[uint](node), align.uint, sizeof(AllocationHeader))
    let requiredSpace = size + padding
    if node.blockSize >= requiredSpace:
      break
    prev = node
    node = node.next
  result = (node, prev, padding)

proc findBest(x: FreeList, size, align: int): tuple[node, prev: FreeNode, padding: int] =
  var
    smallestDiff = high(int)
    node = x.head
    prev: FreeNode = nil
    bestNode: FreeNode = nil
    padding = 0
  while node != nil:
    padding = calcPaddingWithHeader(cast[uint](node), align.uint, sizeof(AllocationHeader))
    let requiredSpace = size + padding
    if node.blockSize >= requiredSpace and (node.blockSize - requiredSpace < smallestDiff):
      bestNode = node
      smallestDiff = node.blockSize - requiredSpace
    prev = node
    node = node.next
  result = (bestNode, prev, padding)

proc alignedAlloc*(fl: var FreeList, size, align: Natural): pointer =
  var
    padding = 0
    prevNode: FreeNode = nil
    node: FreeNode = nil
    alignmentPadding, requiredSpace, remaining: int
    headerPtr: ptr AllocationHeader
  let adjustedSize = max(size, sizeof(FreeNodeObj))
  let adjustedAlign = max(align, DefaultAlignment)
  if fl.policy == FindBest:
    (node, prevNode, padding) = findBest(fl, adjustedSize, adjustedAlign)
  else:
    (node, prevNode, padding) = findFirst(fl, adjustedSize, adjustedAlign)
  if node == nil:
    assert false, "Free list has no free memory"
    return nil
  alignmentPadding = padding - sizeof(AllocationHeader)
  requiredSpace = adjustedSize + padding
  remaining = node.blockSize - requiredSpace
  if remaining > 0:
    let newNode = cast[FreeNode](cast[uint](node) + requiredSpace.uint)
    newNode.blockSize = remaining
    nodeInsert(fl.head, node, newNode)
  nodeRemove(fl.head, prevNode, node)
  headerPtr = cast[ptr AllocationHeader](cast[uint](node) + alignmentPadding.uint)
  headerPtr.blockSize = requiredSpace
  headerPtr.padding = alignmentPadding
  fl.used += requiredSpace
  result = cast[pointer](cast[uint](headerPtr) + sizeof(AllocationHeader).uint)

proc free(fl: var FreeList, p: pointer) =
  if p == nil:
    return
  let header = cast[ptr AllocationHeader](cast[uint](p) - sizeof(AllocationHeader).uint)
  var freeNode = cast[FreeNode](header)
  freeNode.blockSize = header.blockSize + header.padding
  freeNode.next = nil
  var
    node = fl.head
    prevNode: FreeNode = nil
  while node != nil:
    if cast[uint](p) < cast[uint](node):
      nodeInsert(fl.head, prevNode, freeNode)
      break
    prevNode = node
    node = node.next
  fl.used -= freeNode.blockSize
  coalescence(fl, prevNode, freeNode)

proc coalescence(fl: var FreeList, prevNode, freeNode: FreeNode) =
  if freeNode.next != nil and (cast[uint](freeNode) + freeNode.blockSize.uint == cast[uint](freeNode.next)):
    freeNode.blockSize += freeNode.next.blockSize
    nodeRemove(fl.head, freeNode, freeNode.next)
  if prevNode != nil and (prevNode.next != nil) and
      (cast[uint](prevNode) + prevNode.blockSize.uint == cast[uint](freeNode)):
    prevNode.blockSize += freeNode.blockSize
    nodeRemove(fl.head, prevNode, freeNode)

proc nodeInsert(phead: var FreeNode, prevNode, newNode: FreeNode) =
  if prevNode == nil:
    if phead != nil:
      newNode.next = phead
      # phead = newNode
    else:
      phead = newNode
  else:
    if prevNode.next == nil:
      prevNode.next = newNode
      newNode.next = nil
    else:
      newNode.next = prevNode.next
      prevNode.next = newNode

proc nodeRemove(phead: var FreeNode, prevNode, delNode: FreeNode) =
  if prevNode == nil:
    phead = delNode.next
  else:
    prevNode.next = delNode.next
