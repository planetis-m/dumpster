# https://github.com/pavel-kirienko/o1heap
from std/math import isPowerOfTwo
import utils

const
  MaxBins = when sizeof(int) > 2: 32 else: 20
  MemAlign = sizeof(pointer) * 4
  MinChunkSize = MemAlign * 2
  MaxChunkSize = high(int) shr 1 + 1

static:
  assert isPowerOfTwo(MemAlign)
  assert isPowerOfTwo(MinChunkSize)
  assert isPowerOfTwo(MaxChunkSize)

type
  Chunk = object
    header: ChunkHeader
    nextFree: ptr Chunk
    prevFree: ptr Chunk

  ChunkHeader = object
    next: ptr Chunk
    prev: ptr Chunk
    size: int
    used: bool

  FixedHeap = object
    bins: array[MaxBins, ptr Chunk]
    nonEmptyBinMask: uint
    capacity, occupied: int

proc interlink(left, right: ptr Chunk) =
  ## Links two blocks so that their next/prev pointers point to each other; left goes before right.
  if left != nil:
    left.header.next = right
  if right != nil:
    right.header.prev = left

proc addToBin(x: var FixedHeap, b: ptr Chunk) =
  ## Adds a new block into the appropriate bin and updates the lookup mask.
  assert b != nil
  assert b.header.size >= MinChunkSize
  assert b.header.size mod MinChunkSize == 0
  let idx = log2Floor(b.header.size div MinChunkSize)
  assert idx < MaxBins
  # Add the new block to the beginning of the bin list
  b.nextFree = x.bins[idx]
  b.prevFree = nil
  if x.bins[idx] != nil:
    x.bins[idx].prevFree = b
  x.bins[idx] = b
  x.nonEmptyBinMask = x.nonEmptyBinMask or pow2(idx).uint

proc delFromBin(x: var FixedHeap, b: ptr Chunk) =
  ## Removes the specified block from its bin.
  assert b != nil
  assert b.header.size >= MinChunkSize
  assert b.header.size mod MinChunkSize == 0
  let idx = log2Floor(b.header.size div MinChunkSize)
  assert idx < MaxBins
  # Remove the bin from the free block list
  if b.nextFree != nil:
    b.nextFree.prevFree = b.prevFree
  if b.prevFree != nil:
    b.prevFree.nextFree = b.nextFree
  # Update the bin header
  if x.bins[idx] == b:
    assert b.prevFree == nil
    x.bins[idx] = b.nextFree
    if x.bins[idx] == nil:
      x.nonEmptyBinMask = x.nonEmptyBinMask and not pow2(idx).uint

proc createFixedHeap*(buffer: openarray[byte]): FixedHeap =
  result = FixedHeap()
  let base = alignUp(cast[pointer](buffer), MemAlign)
  let padding = cast[uint](base) - cast[uint](base)
  let size = buffer.len - padding.int
  if base != nil and size >= MinChunkSize:
    # Limit and align the capacity
    var capacity = size
    if capacity > MaxChunkSize:
      capacity = MaxChunkSize
    while capacity mod MinChunkSize != 0:
      assert capacity > 0
      dec(capacity)
    assert capacity mod MinChunkSize == 0
    assert capacity >= MinChunkSize and capacity <= MaxChunkSize
    # Initialize the root block
    let b = cast[ptr Chunk](base)
    assert cast[uint](b) mod MemAlign == 0
    b.header.next = nil
    b.header.prev = nil
    b.header.size = capacity
    b.header.used = false
    b.nextFree = nil
    b.prevFree = nil
    addToBin(result, b)
    assert result.nonEmptyBinMask != 0
    result.capacity = capacity

proc alloc(x: var FixedHeap, amount: int): pointer =
  assert x.capacity <= MaxChunkSize
  result = nil
  if amount > 0 and amount <= x.capacity - MemAlign:
    let chunkSize = nextPowerOfTwo(amount + MemAlign)
    assert chunkSize <= MaxChunkSize
    assert chunkSize >= MinChunkSize
    assert chunkSize >= amount + MemAlign
    assert isPowerOfTwo(chunkSize)
    let optimalBinIndex = log2Ceil(chunkSize div MinChunkSize) # Use ceil when fetching
    assert optimalBinIndex < MaxBins
    let candidateBinMask = not (pow2(optimalBinIndex) - 1)
    let suitableBins = x.nonEmptyBinMask and candidateBinMask.uint
    let smallestBinMask = suitableBins and not (suitableBins - 1) # Clear all bits but the lowest
    if smallestBinMask != 0:
      assert isPowerOfTwo(smallestBinMask.int)
      let binIndex = log2Floor(smallestBinMask.int)
      assert binIndex >= optimalBinIndex
      assert binIndex < MaxBins
      let b = x.bins[binIndex]
      assert b != nil
      assert b.header.size >= chunkSize
      assert (b.header.size mod MinChunkSize) == 0
      assert not b.header.used
      delFromBin(x, b)
      let leftover = b.header.size - chunkSize
      b.header.size = chunkSize
      assert leftover < x.capacity # Overflow check
      assert leftover mod MinChunkSize == 0 # Alignment check
      if leftover >= MinChunkSize:
        let newBlock = cast[ptr Chunk](cast[uint](b) + chunkSize.uint)
        assert cast[uint](newBlock) mod MemAlign == 0
        newBlock.header.size = leftover
        newBlock.header.used = false
        interlink(newBlock, b.header.next)
        interlink(b, newBlock)
        addToBin(x, newBlock)
      assert x.occupied mod MinChunkSize == 0
      inc x.occupied, chunkSize
      assert x.occupied <= x.capacity
      assert b.header.size >= amount + MemAlign
      b.header.used = true
      result = cast[pointer](cast[uint](b) + MemAlign)

proc free(x: var FixedHeap, p: pointer) =
  assert x.capacity <= MaxChunkSize
  if p != nil: # nil pointer is a no-op.
    let b = cast[ptr Chunk](cast[uint](p) - MemAlign)
    assert cast[uint](b) mod sizeof(pointer).uint == 0
    assert b.header.used # Catch double-free
    assert cast[uint](b.header.next) mod sizeof(pointer).uint == 0
    assert cast[uint](b.header.prev) mod sizeof(pointer).uint == 0
    assert b.header.size >= MinChunkSize
    assert b.header.size <= x.capacity
    assert b.header.size mod MinChunkSize == 0
    # Even if we're going to drop the block later, mark it free anyway to prevent double-free
    b.header.used = false
    # Update the diagnostics. It must be done before merging because it invalidates the block size information.
    assert x.occupied >= b.header.size # Heap corruption check
    dec x.occupied, b.header.size
    # Merge with siblings and insert the returned block into the appropriate bin and update metadata.
    let prev = b.header.prev
    let next = b.header.next
    let joinLeft = prev != nil and not prev.header.used
    let joinRight = next != nil and not next.header.used
    if joinLeft and joinRight: # [ prev ][ this ][ next ] => [ ------- prev ------- ]
      delFromBin(x, prev)
      delFromBin(x, next)
      inc prev.header.size, b.header.size + next.header.size
      b.header.size = 0 # Invalidate the dropped block headers to prevent double-free.
      next.header.size = 0
      assert prev.header.size mod MinChunkSize == 0
      interlink(prev, next.header.next)
      addToBin(x, prev)
    elif joinLeft: # [ prev ][ this ][ next ] => [ --- prev --- ][ next ]
      delFromBin(x, prev)
      inc prev.header.size, b.header.size
      b.header.size = 0
      assert prev.header.size mod MinChunkSize == 0
      interlink(prev, next)
      addToBin(x, prev)
    elif joinRight: # [ prev ][ this ][ next ] => [ prev ][ --- this --- ]
      delFromBin(x, next)
      inc b.header.size, next.header.size
      next.header.size = 0
      assert b.header.size mod MinChunkSize == 0
      interlink(b, next.header.next)
      addToBin(x, b)
    else:
      addToBin(x, b)

when isMainModule:
  import std/random

  const BufferSize = 1024
  var backingBuffer: array[BufferSize, byte]
  var x = createFixedHeap(backingBuffer)

  block:
    assert x.occupied == 0

    let p1 = x.alloc(100)
    assert p1 != nil
    assert x.occupied > 100

    let p2 = x.alloc(200)
    assert p2 != nil
    assert cast[uint](p2) > cast[uint](p1)

    x.free(p1)

    x.free(p2)
    assert x.occupied == 0

  block: # Edge cases
    let p1 = x.alloc(x.capacity - MinChunkSize)
    assert p1 != nil

    x.free(p1)
    assert x.occupied == 0

    let p2 = x.alloc(x.capacity * 2)
    assert p2 == nil

  block: # Coalescence
    let p1 = x.alloc(100)
    let p2 = x.alloc(100)
    let p3 = x.alloc(100)

    x.free(p2)  # Create a hole
    let usedAfterFree = x.occupied

    x.free(p1)  # Should coalesce with the hole
    assert x.occupied < usedAfterFree

    x.free(p3)  # Should coalesce everything back
    assert x.occupied == 0

  block: # Multiple
    var ptrs: seq[pointer] = @[]
    for i in 1..8:
      let p = x.alloc(i * 10)
      assert p != nil
      ptrs.add(p)

    for p in ptrs:
      x.free(p)

    assert x.occupied == 0

  block:
    let p1 = x.alloc(100)
    let p2 = x.alloc(100)
    x.free(p1)

    let p3 = x.alloc(50)

    assert cast[uint](p3) < cast[uint](p2)

    x.free(p2)
    x.free(p3)

    assert x.occupied == 0

  block: # stress test
    var ptrs: seq[pointer] = @[]
    for i in 1..100:
      let size = rand(1..50)
      let p = x.alloc(size)
      if p != nil:
        ptrs.add(p)

    for i in 0..<ptrs.len:
      if i mod 2 == 0:
        x.free(ptrs[i])

    for i in 0..<ptrs.len:
      if i mod 2 != 0:
        x.free(ptrs[i])

    assert x.occupied == 0

  block:
    let buffer = newSeq[byte](pow2(20)) # fill 13 bins
    var x = createFixedHeap(buffer)
    var allocs: array[1..13, pointer]
    for i, p in allocs.mpairs:
      p = x.alloc(pow2(i+6) - MinChunkSize)
      assert p != nil
      # echo (memory: x.occupied)
      # echo (binmask: toBin(x.nonEmptyBinMask.int, 64))
    for p in allocs:
      x.free(p)
      # echo (binmask: toBin(x.nonEmptyBinMask.int, 64))
      # echo (bins: x.bins)
