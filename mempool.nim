# Fast Efficient Fixed-Size Memory Pool https://arxiv.org/pdf/2210.16471

type
  FixedPool* = object
    numOfBlocks: uint32
    sizeOfEachBlock: uint32
    numFreeBlocks: uint32
    numInitialized: uint32
    memStart: seq[byte]
    next: int

proc createFixedPool*(sizeOfEachBlock: int, numOfBlocks: uint32): FixedPool =
  result = FixedPool(
    numOfBlocks: numOfBlocks,
    sizeOfEachBlock: uint32(sizeOfEachBlock),
    memStart: newSeq[byte](sizeOfEachBlock * numOfBlocks.int),
    numFreeBlocks: numOfBlocks,
    next: 0
  )

proc addrFromIndex(x: FixedPool, i: uint32): int {.inline.} =
  int(i * x.sizeOfEachBlock)

proc indexFromAddr(x: FixedPool, p: int): uint32 {.inline.} =
  uint32(p div int(x.sizeOfEachBlock))

proc alloc*(x: var FixedPool): pointer =
  if x.numInitialized < x.numOfBlocks:
    let p = cast[ptr uint32](addr x.memStart[x.addrFromIndex(x.numInitialized)])
    p[] = x.numInitialized + 1
    inc(x.numInitialized)
  if x.numFreeBlocks > 0:
    let ret = addr x.memStart[x.next]
    dec(x.numFreeBlocks)
    if x.numFreeBlocks != 0:
      x.next = x.addrFromIndex(cast[ptr uint32](ret)[])
    else:
      x.next = -1
    return ret
  return nil

proc dealloc*(x: var FixedPool, p: pointer) =
  let index = cast[int](p) - cast[int](addr x.memStart[0])
  if x.next != -1:
    cast[ptr uint32](p)[] = x.indexFromAddr(x.next)
    x.next = index
  else:
    cast[ptr uint32](p)[] = x.numOfBlocks
    x.next = index
  inc(x.numFreeBlocks)
