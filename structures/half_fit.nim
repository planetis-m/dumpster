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

const
  InstanceSizePadded = alignup(sizeof(FixedHeap).uint, MemAlign).int

proc interlink(left, right: ptr Chunk) =
  ## Links two blocks so that their next/prev pointers point to each other; left goes before right.
  if left != nil:
    left.header.next = right
  if right != nil:
    right.header.prev = left

proc reBin(x: var FixedHeap, b: ptr Chunk) =
  ## Adds a new block into the appropriate bin and updates the lookup mask.
  # assert b != nil
  # assert b.header.size >= MinChunkSize
  # assert (b.header.size mod MinChunkSize) == 0
  let idx = log2Floor(b.header.size div MinChunkSize)
  # assert i < MaxBins
  # Add the new block to the beginning of the bin list
  b.nextFree = x.bins[idx]
  b.prevFree = nil
  if x.bins[idx] != nil:
    x.bins[idx].prevFree = b
  x.bins[idx] = b
  x.nonEmptyBinMask = x.nonEmptyBinMask or pow2(idx.uint)

proc unBin(x: var FixedHeap, b: ptr Chunk) =
  ## Removes the specified block from its bin.
  # assert b != nil
  # assert b.header.size >= MinChunkSize
  # assert (b.header.size mod MinChunkSize) == 0
  let idx = log2Floor(b.header.size div MinChunkSize)
  # assert(i < MaxBins)
  # Remove the bin from the free block list
  if b.nextFree != nil:
    b.nextFree.prevFree = b.prevFree
  if b.prevFree != nil:
    b.prevFree.nextFree = b.nextFree
  # Update the bin header
  if x.bins[idx] == b:
    # assert b.prevFree == nil
    x.bins[idx] = b.nextFree
    if x.bins[idx] == nil:
      x.nonEmptyBinMask = x.nonEmptyBinMask and not pow2(idx.uint)

proc createFixedHeap*(buffer: openarray[byte]): ptr FixedHeap =
  result = nil
  let base = cast[pointer](buffer)
  let size = buffer.len
  if base != nil and cast[uint](base) mod MemAlign == 0 and
      size >= InstanceSizePadded + MinChunkSize:
    # Allocate the core heap metadata structure in the beginning of the arena
    result = cast[ptr FixedHeap](base)
    result.nonEmptyBinMask = 0
    for i in 0 ..< MaxBins:
      result.bins[i] = nil
    # Limit and align the capacity
    var capacity = size - InstanceSizePadded
    if capacity > MaxChunkSize:
      capacity = MaxChunkSize
    while capacity mod MinChunkSize != 0:
      # assert capacity > 0
      dec(capacity)
    # assert capacity mod MinChunkSize == 0
    # assert (capacity >= MinChunkSize) and (capacity <= MaxChunkSize)
    # Initialize the root block
    let b = cast[ptr Chunk](cast[uint](base) + InstanceSizePadded.uint)
    # assert cast[uint](b) mod MemAlign == 0
    b.header.next = nil
    b.header.prev = nil
    b.header.size = capacity
    b.header.used = false
    b.nextFree = nil
    b.prevFree = nil
    reBin(result[], b)
    # assert result.nonEmptyBinMask != 0

proc alloc(x: var FixedHeap, amount: int): pointer =
  # assert x.capacity <= MaxChunkSize)
  result = nil
  if amount > 0 and amount <= x.capacity - MemAlign:
    let chunkSize = nextPowerOfTwo(amount + MemAlign)
    # assert chunkSize <= MaxChunkSize
    # assert chunkSize >= MinChunkSize
    # assert chunkSize >= amount + MemAlign
    # assert isPowerOfTwo(chunkSize)

    let optimalBinIndex = log2Ceil(chunkSize div MinChunkSize) # Use ceil when fetching.
    # assert optimalBinIndex < MaxBins
    let candidateBinMask = not (pow2(optimalBinIndex.uint) - 1)

    let suitableBins = x.nonEmptyBinMask and candidateBinMask
    let smallestBinMask = suitableBins and not (suitableBins - 1)  # Clear all bits but the lowest.
    if smallestBinMask != 0:
      # assert isPowerOfTwo(smallestBinMask)
      let binIndex = log2Floor(smallestBinMask.int)
      # assert binIndex >= optimalBinIndex
      # assert binIndex < MaxBins
      let b = x.bins[binIndex]
      # assert b != nil
      # assert b.header.size >= chunkSize
      # assert (b.header.size mod MinChunkSize) == 0
      # assert not b.header.used)
      unBin(x, b)
      let leftover = b.header.size - chunkSize
      b.header.size = chunkSize
      # assert leftover < x.capacity # Overflow check.
      # assert leftover mod MinChunkSize == 0 # Alignment check.
      if leftover >= MinChunkSize:
        let newBlock = cast[ptr Chunk](cast[uint](b) + chunkSize.uint)
        # assert cast[uint](newBlock) mod MemAlign == 0)
        newBlock.header.size = leftover
        newBlock.header.used = false
        interlink(newBlock, b.header.next)
        interlink(b, newBlock)
        reBin(x, newBlock)
      # assert (x.occ mod MinChunkSize) == 0)
      x.occ += chunkSize
      # assert x.occ <= x.capacity
      # if x.maxMem < x.occ:
      #   x.maxMem = x.occ
      # assert b.header.size >= amount + MemAlign)
      b.header.used = true
      result = cast[pointer](cast[uint](b) + MemAlign)

proc free(x: var FixedHeap, p: pointer) =
  # assert x.capacity <= MaxChunkSize
  if p != nil:  # NULL pointer is a no-op.
    let b = cast[ptr Chunk](cast[uint](p) - MemAlign)
    # Check for heap corruption in debug builds.
    # assert cast[uint](b) mod sizeof(pointer) == 0
    # assert cast[uint](b) >= (cast[uint](addr x) + InstanceSizePadded)
    # assert cast[uint](b) <=
    #   (cast[uint](addr x) + InstanceSizePadded + x.capacity - MinChunkSize)
    # assert b.header.used # Catch double-free
    # assert cast[uint](b.header.next) mod sizeof(pointer) == 0
    # assert cast[uint](b.header.prev) mod sizeof(pointer) == 0
    # assert b.header.size >= MinChunkSize
    # assert b.header.size <= x.capacity
    # assert (b.header.size mod MinChunkSize) == 0
    # Even if we're going to drop the block later, mark it free anyway to prevent double-free.
    b.header.used = false
    # Update the diagnostics. It must be done before merging because it invalidates the block size information.
    # assert x.occ >= b.header.size # Heap corruption check.
    x.occ -= b.header.size
    # Merge with siblings and insert the returned block into the appropriate bin and update metadata.
    let prev = b.header.prev
    let next = b.header.next
    let joinLeft = (prev != nil) and (not prev.header.used)
    let joinRight = (next != nil) and (not next.header.used)
    if joinLeft and joinRight:  # [ prev ][ this ][ next ] => [ ------- prev ------- ]
      unBin(x, prev)
      unBin(x, next)
      prev.header.size += b.header.size + next.header.size
      b.header.size = 0  # Invalidate the dropped block headers to prevent double-free.
      next.header.size = 0
      # assert (prev.header.size mod MinChunkSize) == 0
      interlink(prev, next.header.next)
      reBin(x, prev)
    elif joinLeft:  # [ prev ][ this ][ next ] => [ --- prev --- ][ next ]
      unBin(x, prev)
      prev.header.size += b.header.size
      b.header.size = 0
      # assert (prev.header.size mod MinChunkSize) == 0
      interlink(prev, next)
      reBin(x, prev)
    elif joinRight:  # [ prev ][ this ][ next ] => [ prev ][ --- this --- ]
      unBin(x, next)
      b.header.size += next.header.size
      next.header.size = 0
      # assert (b.header.size mod MinChunkSize) == 0
      interlink(b, next.header.next)
      reBin(x, b)
    else:
      reBin(x, b)
