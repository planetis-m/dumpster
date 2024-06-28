# https://github.com/pavel-kirienko/o1heap
from std/math import isPowerOfTwo
import utils

const
  MaxBins = when sizeof(int) > 2: 32 else: 16
  MemAlign = sizeof(pointer) * 4
  MinChunkSize = MemAlign * 2
  MaxChunkSize = (high(int) shr 1) + 1

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
    capacity, occ: int

proc interlink(left, right: ptr Chunk) =
  ## Links two blocks so that their next/prev pointers point to each other; left goes before right.
  if left != nil:
    left.header.next = right
  if right != nil:
    right.header.prev = left

proc addToBin(x: var FixedHeap, b: ptr Chunk) =
  ## Adds a new block into the appropriate bin and updates the lookup mask.
  let idx = log2Floor(b.header.size div MinChunkSize)
  # Add the new block to the beginning of the bin list
  b.nextFree = x.bins[idx]
  b.prevFree = nil
  if x.bins[idx] != nil:
    x.bins[idx].prevFree = b
  x.bins[idx] = b
  x.nonEmptyBinMask = x.nonEmptyBinMask or pow2(idx.uint)

proc delFromBin(x: var FixedHeap, b: ptr Chunk) =
  ## Removes the specified block from its bin.
  let idx = log2Floor(b.header.size div MinChunkSize)
  # Remove the bin from the free block list
  if b.nextFree != nil:
    b.nextFree.prevFree = b.prevFree
  if b.prevFree != nil:
    b.prevFree.nextFree = b.nextFree
  # Update the bin header
  if x.bins[idx] == b:
    x.bins[idx] = b.nextFree
    if x.bins[idx] == nil:
      x.nonEmptyBinMask = x.nonEmptyBinMask and not pow2(idx.uint)

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
      dec(capacity)
    # Initialize the root block
    let b = cast[ptr Chunk](base)
    b.header.next = nil
    b.header.prev = nil
    b.header.size = capacity
    b.header.used = false
    b.nextFree = nil
    b.prevFree = nil
    addToBin(result, b)

proc alloc(x: var FixedHeap, amount: int): pointer =
  result = nil
  if amount > 0 and amount <= x.capacity - MemAlign:
    let chunkSize = nextPowerOfTwo(amount + MemAlign)
    let optimalBinIndex = log2Ceil(chunkSize div MinChunkSize) # Use ceil when fetching.
    let candidateBinMask = not (pow2(optimalBinIndex.uint) - 1)
    let suitableBins = x.nonEmptyBinMask and candidateBinMask
    let smallestBinMask = suitableBins and not (suitableBins - 1) # Clear all bits but the lowest.
    if smallestBinMask != 0:
      let binIndex = log2Floor(smallestBinMask.int)
      let b = x.bins[binIndex]
      delFromBin(x, b)
      let leftover = b.header.size - chunkSize
      b.header.size = chunkSize
      if leftover >= MinChunkSize:
        let newBlock = cast[ptr Chunk](cast[uint](b) + chunkSize.uint)
        newBlock.header.size = leftover
        newBlock.header.used = false
        interlink(newBlock, b.header.next)
        interlink(b, newBlock)
        addToBin(x, newBlock)
      inc x.occ, chunkSize
      b.header.used = true
      result = cast[pointer](cast[uint](b) + MemAlign)

proc free(x: var FixedHeap, p: pointer) =
  if p != nil: # nil pointer is a no-op.
    let b = cast[ptr Chunk](cast[uint](p) - MemAlign)
    # Even if we're going to drop the block later, mark it free anyway to prevent double-free.
    b.header.used = false
    # Update the diagnostics. It must be done before merging because it invalidates the block size information.
    dec x.occ, b.header.size
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
      interlink(prev, next.header.next)
      addToBin(x, prev)
    elif joinLeft: # [ prev ][ this ][ next ] => [ --- prev --- ][ next ]
      delFromBin(x, prev)
      inc prev.header.size, b.header.size
      b.header.size = 0
      interlink(prev, next)
      addToBin(x, prev)
    elif joinRight: # [ prev ][ this ][ next ] => [ prev ][ --- this --- ]
      delFromBin(x, next)
      inc b.header.size, next.header.size
      next.header.size = 0
      interlink(b, next.header.next)
      addToBin(x, b)
    else:
      addToBin(x, b)
