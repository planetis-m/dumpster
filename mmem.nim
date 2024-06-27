# https://youtu.be/UTii4dyhR5c
# https://github.com/JiajunWan/malloc_lab
# Basic constants

const
  WSize = sizeof(uint) # Word and header size (bytes)
  DSize = 2 * WSize # Double word size (bytes)
  MinBlockSize = DSize # Minimum block size (bytes)

  # Expand heap by at least chunksize (4096) each time no free space (Must be
  # divisible by dsize)
  ChunkSize = 1 shl 12
  InitSize = 1 shl 6 # Heap init size

  AllocMask = 0x1'u # Mask to get the alloc bit
  SizeMask = not 0xf'u # Mask to get the size

  LowerBitsMask = 0xf'u # Lower four bits mask 1111
  AllocPrevMask = 0x2'u # Prev alloc mask 0010
  PrevMiniMask = 0x4'u # Prev mini size mask 0100
  AllocPrevMiniMask = 0x6'u # Prev alloc mini size mask 0110
  AllocMiniMask = 0x8'u # Current block mini mask 1000
  AllocMiniAllocPrev = 0xA'u # Current block is mini block, prev alloc mask 1010
  AddrMask = not 0x7'u # Address mask 1111111....0000 ???

  SegCount = 15 # Seg list buckets count
  SegSize: array[SegCount, int] = [ # Seg list size const array
    16,   32,   64,   96,    128,   256,   512,      1024,
    2048, 4096, 8192, 16384, 32768, 65536, high(int64)
  ]

type
  Chunk = object ## Represents the header and payload of one block in the heap
    # Header contains size + allocation flag
    header: uint
    payload: UncheckedArray[byte]

  Free = object ## Free alias struct for payload area usage
    next: ptr Chunk
    prev: ptr Chunk
    payload: UncheckedArray[byte]

# Global variables

var
  # Explicit free list root pointer, lowest address, insert starting point
  segListRoot: array[SegCount, ptr Chunk]
  heapStart: ptr Chunk = nil # Pointer to first block

proc roundup(size, n: int): int =
  ## roundup: Rounds size up to next multiple of n
  n * ((size + (n - 1)) div n)

proc pack(size: int, alloc: bool, allocPrev: uint): uint =
  ## pack: returns a header reflecting a specified size and its alloc status.
  ##       If the block is allocated, the lowest bit is set to 1, and 0 otherwise.
  if alloc: uint(size) or AllocMask or allocPrev
  else: uint(size) or allocPrev

proc extractSize(word: uint): int =
  ## extractSize: returns the size of a given header value based on the header
  ##              specification above.
  int(word and SizeMask)

proc getSize(b: ptr Chunk): int =
  ## getSize: returns the size of a given block by clearing the lowest 4 bits
  ##          (as the heap is 16-byte aligned).
  extractSize(b.header)

proc getPayloadSize(b: ptr Chunk): int =
  ## getPayloadSize: returns the payload size of a given block, equal to
  ##                 the entire block size minus the header and footer sizes.
  let asize = getSize(b)
  asize - WSize

proc extractAlloc(word: uint): bool =
  ## extractAlloc: returns the allocation status of a given header value based
  ##               on the header specification above.
  bool(word and AllocMask)

proc getAlloc(b: ptr Chunk): bool =
  ## getAlloc: returns true when the block is allocated based on the
  ##           block header's lowest bit, and false otherwise.
  extractAlloc(b.header)

proc getPrevAlloc(b: ptr Chunk): bool =
  ## Get the prev alloc bit from current block
  bool(b.header and AllocPrevMask)

proc payloadToHeader(p: pointer): ptr Chunk =
  ## payloadToHeader: given a payload pointer, returns a pointer to the
  ##                  corresponding block.
  result = cast[ptr Chunk](cast[uint](p) - offsetOf(Chunk, payload).uint)

proc headerToPayload(b: ptr Chunk): pointer =
  ## headerToPayload: given a b pointer, returns a pointer to the
  ##                  corresponding payload.
  result = addr(b.payload)

proc headerToFooter(b: ptr Chunk): ptr uint =
  ## headerToFooter: given a block pointer, returns a pointer to the
  ##                 corresponding footer.
  result = cast[ptr uint](cast[uint](b.payload.addr) + getSize(b).uint - DSize)

proc writeHeader(b: ptr Chunk, size: int, alloc: bool, allocPrev: uint) =
  ## writeHeader: given a block and its size and allocation status,
  ##              writes an appropriate value to the block header.
  ## Pre: Chunk is a valid block pointer, size is non-negative
  ## Post: The header is written based on the given size and lower bits
  assert b != nil
  b.header = pack(size, alloc, allocPrev)

proc writeFooter(b: ptr Chunk, size: int, alloc: bool, allocPrev: uint) =
  ## writeFooter: given a block and its size and allocation status,
  ##              writes an appropriate value to the block footer by first
  ##              computing the position of the footer.
  ## Pre: Chunk is a valid block pointer, size is non-negative
  ## Post: The footer is written based on the given size and lower bits
  assert b != nil
  assert getSize(b) == size and size > 0
  let footerp = headerToFooter(b)
  footerp[] = pack(size, alloc, allocPrev)

proc findNext(b: ptr Chunk): ptr Chunk =
  ## findNext: returns the next consecutive block on the heap by adding the
  ##           size of the block.
  assert b != nil
  assert getSize(b) != 0
  result = cast[ptr Chunk](cast[uint](b) + getSize(b).uint)

proc findPrevFooter(b: ptr Chunk): ptr uint =
  ## findPrevFooter: returns the footer of the previous block.
  # Compute previous footer position as one word before the header
  result = cast[ptr uint](cast[uint](b.header.addr) - WSize.uint)

proc findPrev(b: ptr Chunk): ptr Chunk =
  ## findPrev: returns the previous block position by checking the previous
  ##           block's footer and calculating the start of the previous block
  ##           based on its size.
  assert b != nil
  assert getSize(b) != 0
  let footerp = findPrevFooter(b)
  let size = extractSize(footerp[])
  result = cast[ptr Chunk](cast[uint](b) - size.uint)

proc removeBlock(b: ptr Chunk, root: ptr ptr Chunk, isMini: bool) =
  ## Remove a block from the free list pointed by the root. Do a different
  ## mini block remove if the block is a mini block
  if not isMini:
    # Case 1: Only one block
    if root[] == cast[ptr Free](root[].payload).next:
      root[] = nil
    # Case 2: Block is root
    elif b == root[]:
      let tail = cast[ptr Free](root[].payload).prev
      root[] = cast[ptr Free](b.payload).next
      cast[ptr Free](root[].payload).prev = tail
      cast[ptr Free](tail.payload).next = root[]
    # Case 3: Block is else where
    else:
      let blockNext = cast[ptr Free](b.payload).next
      let blockPrev = cast[ptr Free](b.payload).prev

      cast[ptr Free](blockPrev.payload).next = blockNext
      cast[ptr Free](blockNext.payload).prev = blockPrev
  # Mini seg list
  else:
    # Case 1: Only one b
    if root[] == cast[ptr Chunk](root[].header and AddrMask):
      root[] = nil
    # Case 2: Block is root
    elif b == root[]:
      let tail = cast[ptr Chunk](cast[ptr uint](root[].payload)[] and AddrMask)
      let tailAddr = cast[uint](tail) and AddrMask
      let tailLowerBits = tail.header and LowerBitsMask
      root[] = cast[ptr Chunk](uint(b.header) and AddrMask)
      let rootAddr = cast[uint](root[]) and AddrMask
      let rootLowerBits = root[].header and LowerBitsMask

      cast[ptr uint](root[].payload)[] = tailAddr or rootLowerBits
      cast[ptr uint](tail)[] = rootAddr or tailLowerBits
    # Case 3: Block is else where
    else:
      let blockNext = cast[ptr Chunk](b.header and AddrMask)
      let nextAddr = cast[uint](blockNext) and AddrMask
      let nextLowerBits = blockNext.header and LowerBitsMask
      let blockPrev = cast[ptr Chunk](cast[ptr uint](b.payload)[] and AddrMask)
      let prevAddr = cast[uint](blockPrev) and AddrMask
      let prevLowerBits = blockPrev.header and LowerBitsMask

      cast[ptr uint](blockPrev)[] = nextAddr or prevLowerBits
      cast[ptr uint](blockNext.payload)[] = prevAddr or nextLowerBits

proc insertBlockBefore(b: ptr Chunk, blockNext: ptr ptr Chunk, isRoot, isMini: bool) =
  ## LIFO Insert b before the b pointed by blockNext. Update root
  ## if next b is root. Do a different mini b insertion if the
  ## b is a mini b
  # Not mini seg list
  if not isMini:
    # Case 1: No next b, empty list
    if blockNext[] == nil:
      blockNext[] = b
      cast[ptr Free](b.payload).next = b
      cast[ptr Free](b.payload).prev = b
    # Case 2: Not empty free list
    else:
      let blockPrev = cast[ptr Free](blockNext[].payload).prev
      cast[ptr Free](blockPrev.payload).next = b
      cast[ptr Free](b.payload).next = blockNext[]
      cast[ptr Free](blockNext[].payload).prev = b
      cast[ptr Free](b.payload).prev = blockPrev
      # Update the root if the next block is root
      if isRoot:
        blockNext[] = b
  # Mini seg list
  else:
    # Case 1: No next block, empty list
    if blockNext[] == nil:
      let blockAddr = cast[uint](b) and AddrMask
      let lowerBits = b.header and LowerBitsMask
      blockNext[] = b
      cast[ptr uint](b)[] = blockAddr or lowerBits
      cast[ptr uint](b.payload)[] = blockAddr or lowerBits
    # Case 2: Non empty free list
    else:
      let blockAddr = cast[uint](b) and AddrMask
      let lowerBits = b.header and LowerBitsMask
      let blockPrev = cast[ptr Chunk](cast[ptr uint](blockNext[].payload)[] and AddrMask)
      let prevAddr = cast[uint](blockPrev) and AddrMask
      let prevLowerBits = blockPrev.header and LowerBitsMask
      let nextAddr = cast[uint](blockNext[]) and AddrMask
      let nextLowerBits = blockNext[].header and LowerBitsMask

      cast[ptr uint](blockPrev)[] = blockAddr or prevLowerBits
      cast[ptr uint](b)[] = nextAddr or lowerBits
      cast[ptr uint](blockNext[].payload)[] = blockAddr or nextLowerBits
      cast[ptr uint](b.payload)[] = prevAddr or lowerBits
      # Update root if next b is root
      if isRoot:
        blockNext[] = b

proc sizeToRoot(asize: int): ptr ptr Chunk =
  ## Return the pointer to the root of seg free list pointer that adjusted
  ## size is in the size range
  for i in 0..<SegCount:
    if asize <= SegSize[i]:
      return addr(segListRoot[i])
  return addr(segListRoot[SegCount])

proc printHeap() =
  var b = heapStart
  var size: uint
  var alloc: bool
  var allocNext: bool
  var numFreeBlock: uint = 0
  # High epilogue address as 7 bytes backward from last byte
  let high = cast[pointer](cast[uint](memHeapHi()) - 7)
  while true:
    if (b.header and AllocMiniMask) != 0:
      size = DSize
      alloc = getAlloc(b)
      allocNext = getAlloc(cast[ptr Chunk](cast[uint](b) + 2 * uint(sizeof(Chunk))))
    else:
      size = getSize(b)
      alloc = getAlloc(b)
      allocNext = getAlloc(findNext(b))
    if alloc:
      echo &"alloc: 0x{cast[uint](addr b.header):011x}: 0x{b.header:011x} {size:<7}"
    if not alloc:
      inc numFreeBlock
      if (b.header and AllocMiniMask) != 0:
        echo &"free : 0x{cast[uint](addr b.header):011x}: 0x{b.header:011x} {size:<7} next: 0x{uint(b.header):011x} prev: 0x{cast[uint](b.payload):011x}"
      else:
        let freeBlock = cast[ptr Free](b.payload)
        echo &"free : 0x{cast[uint](addr b.header):011x}: 0x{b.header:011x} {size:<7} next: 0x{cast[uint](freeBlock.next):011x} prev: 0x{cast[uint](freeBlock.prev):011x}"
    if (b.header and AllocMiniMask) != 0:
      b = cast[ptr Chunk](cast[uint](b) + 2 * uint(sizeof(Chunk)))
    else:
      b = findNext(b)
    if b == cast[ptr Chunk](high):
      break
  # Print epilogue
  echo &"alloc: 0x{cast[uint](addr b.header):011x}: 0x{b.header:011x} {size:<7}"
  echo &"Free blocks: {numFreeBlock}"

# The remaining content below are helper and debug routines

proc extendHeap(size: int): ptr Chunk =
  ## Extend the heap from the end
  ## Get more free space
  ## Argv: Size of heap to extend
  ## Return the block pointer to the new free space,
  ## coalesced if free block before extended area
  ## Pre: No free block at least size bytes
  ## Post: Free block at least size bytes is at the
  ## end of the heap
  var bp: pointer
  # Allocate an even number of words to maintain alignment
  let roundedSize = roundup(size, DSize)
  bp = memSbrk(roundedSize)
  if bp == cast[pointer](-1):
    return nil
  # Initialize free block header/footer
  # The bp returned by memSbrk as the payload pointer. Payload to header will
  # lead to the previous epilogue. Overwrite this epilogue as the new header.
  var b = cast[ptr Chunk](cast[uint](bp) - sizeof(Chunk))
  # Write the block as free block and keep the prev mini alloc bits
  let allocPrev = b.header and AllocPrevMiniMask
  writeHeader(b, roundedSize, false, allocPrev)
  writeFooter(b, roundedSize, false, allocPrev)
  # Create new epilogue header
  var blockEpilogue = findNext(b)
  # Alloc prev is zero, new extended free block, just normal write header
  writeHeader(blockEpilogue, 0, true, 0)
  # Coalesce in case the previous block was free
  result = coalesceBlock(b, false)

proc findFit(asize: int): ptr Chunk =
  ## Find a free block for malloc
  ## Argv: Adjusted size of the malloc request
  ## Return the found free block or nil if not found
  ## Pre: asize is double word aligned
  ## Post: Return the found free block or nil if not found
  var
    b: ptr Chunk = nil
    blockBest: ptr Chunk = nil
    size = 0
    sizeBest = 0
    n = 0  # Segregated list index
    timeout = 4
    i = 0
    foundFit = false
  # Find the correct size bucket
  # Increase n until found correct size or the end
  while n < SegCount and asize > SegSize[n]:
    inc n
  # Traverse the free list
  while n < SegCount and not foundFit:
    b = segListRoot[n]
    if b != nil:
      while true:
        if n == 0:
          # Find mini block fit
          if asize <= MinBlockSize:
            return segListRoot[n]
          break
        else:
          size = getSize(b)
          # Find best (better) fit
          if asize <= size:
            if not foundFit:
              blockBest = b
              sizeBest = size
              foundFit = true
            if size <= sizeBest:
              blockBest = b
              sizeBest = size
            inc i
        b = cast[ptr Free](b.payload).next
        if i >= timeout or b == segListRoot[n]:
          break
    # Increase n to next seg list
    inc n
  result = blockBest

proc coalesceBlock(b: ptr Chunk, isMini: bool): ptr Chunk =
  ## Coalesce adjacent free blocks
  ## Argv: Block pointer and if it is mini block
  ## Return the block pointer to the coalesced block
  ## Pre: Block is free block
  ## Post: Coalesced block is free block and not a mini block
  ## if coalescing happened
  assert not getAlloc(b)

  var size: int
  # Consecutive next block, not free list next block
  var blockNext: ptr Chunk
  if isMini:
    blockNext = cast[ptr Chunk](cast[uint](b) + 2 * sizeof(Chunk))
    size = DSize
  else:
    blockNext = findNext(b)
    size = getSize(b)
  let prevAlloc = getPrevAlloc(b) # Get header prev alloc bit
  let nextAlloc = getAlloc(blockNext) # Get header alloc bit
  if prevAlloc and nextAlloc: # Prev and next both alloc
    # Update alloc prev bit of next block to zero (free)
    blockNext.header = blockNext.header and not AllocPrevMask
    # Update the prev mini bit for next block
    if isMini:
      blockNext.header = blockNext.header or PrevMiniMask
    # LIFO seg size insertion
    insertBlockBefore(b, sizeToRoot(size), true, isMini)
  # Prev alloc and next free
  elif prevAlloc and not nextAlloc:
    # Update the size of new free block
    let nextMini = (blockNext.header and AllocMiniMask) != 0
    # Get the size of next block
    var nextSize: int
    if nextMini:
      nextSize = DSize
    else:
      nextSize = getSize(blockNext)
    # Remove next block from list
    removeBlock(blockNext, sizeToRoot(nextSize), nextMini)
    size += nextSize
    # Update header and footer
    let allocPrev = b.header and AllocPrevMiniMask
    writeHeader(b, size, false, allocPrev)
    writeFooter(b, size, false, allocPrev)
    let blockNextNext = findNext(b)
    # Change the prev mini of next block to zero
    blockNextNext.header = blockNextNext.header and not PrevMiniMask
    # LIFO seg size insertion
    insertBlockBefore(b, sizeToRoot(size), true, false)
  # Prev free and next alloc
  elif not prevAlloc and nextAlloc:
    let prevMini = (b.header and PrevMiniMask) != 0
    # Consecutive prev block, not free list prev block ??
    var blockPrev: ptr Chunk
    var prevSize: int
    if prevMini:
      blockPrev = cast[ptr Chunk](cast[uint](b) - 2 * sizeof(Chunk))
      prevSize = DSize
    else:
      blockPrev = findPrev(b)
      prevSize = getSize(blockPrev)
    # LIFO remove
    removeBlock(blockPrev, sizeToRoot(prevSize), prevMini)
    size += prevSize
    # Update header and footer
    let allocPrev = blockPrev.header and AllocPrevMiniMask
    writeHeader(blockPrev, size, false, allocPrev)
    writeFooter(blockPrev, size, false, allocPrev)
    # Update alloc prev bit of next block to zero (free)
    blockNext.header = blockNext.header and not AllocPrevMask
    # Change the prev mini of next block to zero
    blockNext.header = blockNext.header and not PrevMiniMask
    b = blockPrev
    # LIFO seg size insertion
    insertBlockBefore(b, sizeToRoot(size), true, false)
  # Next and prev block are both free
  else:
    let prevMini = (b.header and PrevMiniMask) != 0
    var blockPrev: ptr Chunk
    # Get the size of prev block
    var prevSize: int
    if prevMini:
      blockPrev = cast[ptr Chunk](cast[uint](b) - 2 * sizeof(Block))
      prevSize = dsize
    else:
      blockPrev = findPrev(b)
      prevSize = getSize(blockPrev)
    let nextMini = (blockNext.header and AllocMiniMask) != 0
    # Get the size of next block
    var nextSize: int
    if nextMini:
      nextSize = dsize
    else:
      nextSize = getSize(blockNext)
    # LIFO remove next and prev free blocks
    removeBlock(blockNext, sizeToRoot(nextSize), nextMini)
    removeBlock(blockPrev, sizeToRoot(prevSize), prevMini)
    size += nextSize + prevSize
    # Update header and footer
    let allocPrev = blockPrev.header and AllocPrevMiniMask
    writeHeader(blockPrev, size, false, allocPrev)
    writeFooter(blockPrev, size, false, allocPrev)
    b = blockPrev
    let blockNextNext = findNext(blockPrev)
    # Change the prev mini of next block to zero
    blockNextNext.header = blockNextNext.header and not PrevMiniMask
    # LIFO seg size insertion
    insertBlockBefore(b, sizeToRoot(size), true, false)
  assert not getAlloc(b)
  result = b

proc splitBlock(b: ptr Chunk, asize: int) =
  ## Split the allocated block if too big
  ## Argv: Block pointer and adjusted size
  ## Does not return
  ## Pre: Block is marked as allocated,
  ## adjusted size is smaller than the block size
  ## Post: the Block is split if enough free space,
  ## otherwise the block is allocated fully
  assert getAlloc(b)
  let blockSize = getSize(b)
  # Free space left
  let leftSize = blockSize - asize
  # Remove split block
  removeBlock(b, sizeToRoot(blockSize), false)
  # Check if the allocated block is mini block
  let leftMini = asize == minBlockSize
  # Split if there is enough free space
  var blockNew: ptr Chunk = nil
  # Free space is not mini block
  if leftSize > minBlockSize:
    let allocPrev = b.header and allocPrevMiniMask
    # Allocated block is mini block
    if leftMini:
      writeHeader(b, asize, true, allocPrev or AllocMiniMask)
      # Split new block
      blockNew = findNext(b)
      writeHeader(blockNew, leftSize, false, AllocPrevMiniMask)
      writeFooter(blockNew, leftSize, false, AllocPrevMiniMask)
    # Allocated block is not mini block
    else:
      writeHeader(b, asize, true, allocPrev)
      # Split new block
      blockNew = findNext(b)
      writeHeader(blockNew, leftSize, false, AllocPrevMask)
      writeFooter(blockNew, leftSize, false, AllocPrevMask)
    # Free space block
    let blockNext = findNext(blockNew)
    # Change the prev alloc of next block to zero
    blockNext.header = blockNext.header and not allocPrevMask
    # LIFO seg size insertion
    insertBlockBefore(blockNew, sizeToRoot(leftSize), true, false)
  # Split block is mini block
  elif leftSize == minBlockSize:
    let allocPrev = b.header and allocPrevMiniMask
    # Allocated block is mini block
    if leftMini:
      writeHeader(b, asize, true, allocPrev or allocMiniMask)
      # Split new block is mini block
      blockNew = findNext(b)
      writeHeader(blockNew, leftSize, false,
                  allocPrevMiniMask or allocMiniMask)
    # Allocated block is not mini block
    else:
      writeHeader(b, asize, true, allocPrev)
      # Split new block is mini block
      blockNew = findNext(b)
      writeHeader(blockNew, leftSize, false, allocMiniAllocPrev)
    # Get free space mini block
    let blockNext = cast[ptr Chunk](cast[uint](blockNew) + 2 * sizeof(Block))
    # Change the prev alloc of next block to zero and prev mini to one
    blockNext.header = blockNext.header and not allocPrevMask
    blockNext.header = blockNext.header or prevMiniMask
    # LIFO seg size insertion
    insertBlockBefore(blockNew, sizeToRoot(leftSize), true, true)
  assert getAlloc(b)

proc mmInit(): bool =
  ## Initialize the prologue, epilogue, and global variables for the heap
  ## Return if init succeeded
  ## Pre: Empty heap
  ## Post: heap of initsize
  # Create the initial empty heap
  let start = cast[ptr UncheckedArray[uint]](memSbrk(2 * WSize))
  if cast[int](start) == -1:
    return false
  # Heap prologue and epilogue preventing heap boundary coalescing
  start[0] = pack(0, true, AllocPrevMask) # Heap prologue (block footer)
  start[1] = pack(0, true, AllocPrevMask) # Heap epilogue (block header)
  # Heap starts with first "block header", currently the epilogue
  heapStart = cast[ptr Chunk](addr start[1])
  # Init seg list
  for i in 0..<SegCount:
    segListRoot[i] = nil
  # Extend the empty heap with a free block of chunksize bytes
  if extendHeap(InitSize) == nil:
    return false
  return true

proc malloc(size: int): pointer =
  ## Dynamic memory allocation
  ## Allocation size as argument
  ## Return pointer to allocated payload area,
  ## or return nil on FAIL
  ## Argv: allocation size
  ## Pre: size is non-negative
  ## Post: Heap is allocated with the request
  assert mmCheckheap(instantiationInfo().line)
  var asize: int      # Adjusted block size
  var extendsize: int # Amount to extend heap if no fit is found
  var b: ptr Chunk    # Block to allocate
  var bp: pointer = nil
  # Initialize heap if it isn't initialized
  if heapStart == nil:
    discard mmInit()
  # Ignore spurious request
  if size == 0:
    assert mmCheckheap(instantiationInfo().line)
    return bp
  # Adjust block size to include overhead and to meet alignment requirements
  asize = roundup(size + WSize, DSize)
  # Search the free list for a fit
  b = findFit(asize)
  # If no fit is found, request more memory, and then and place the block
  if b == nil:
    # Always request at least chunksize
    extendsize = max(asize, chunksize)
    b = extendHeap(extendsize)
    # Extend_heap returns an error
    if b == nil:
      return bp
  # Check if find fit block is a mini block
  let blockMini = (b.header and AllocMiniMask) != 0
  # The block should be marked as free
  assert not getAlloc(b)
  # Mark block as allocated
  # If mini block
  if blockMini:
    # First remove the mini block from seg list
    removeBlock(b, sizeToRoot(asize), blockMini)
    # Get the prev mini and alloc bits
    let allocPrev = b.header and AllocPrevMiniMask
    # Write the block header as alloc mini block and keep the middle prev two bits
    writeHeader(b, dsize, true, allocPrev or AllocMiniMask)
    let blockNext = cast[ptr Chunk](cast[uint](b) + 2 * uint(sizeof(Chunk)))
    # Write alloc mini prev bits of next block
    blockNext.header = blockNext.header or AllocPrevMiniMask
  else:
    let blockSize = getSize(b)
    # Get the prev mini and alloc bits
    let allocPrev = b.header and AllocPrevMiniMask
    # Write the block header as alloc block and keep the middle prev two bits
    writeHeader(b, blockSize, true, allocPrev)
    # Write alloc prev of next block
    let blockNext = findNext(b)
    blockNext.header = blockNext.header or AllocPrevMask
    # Try to split the block if too large
    splitBlock(b, asize)
  bp = headerToPayload(b)
  assert mmCheckheap(instantiationInfo().line)
  return bp

proc free(bp: pointer) =
  ## Give back the allocated space to the free space
  ## Argv: payload pointer bp to the starting address of payload
  ## Does not return
  ## Pre: bp payload pointer is pointing to a allocated space
  ## Post: The allocated space is freed
  assert mmCheckheap(instantiationInfo().line)
  if bp == nil:
    return
  var b = payloadToHeader(bp)
  var size: int
  let mini = (b.header and AllocMiniMask) != 0
  if mini:
    size = DSize
  else:
    size = getSize(b)
  # The block should be marked as allocated
  assert getAlloc(b)
  let allocPrev = b.header and AllocPrevMiniMask
  # Mark the block as free
  if mini:
    writeHeader(b, size, false, allocPrev or AllocMiniMask)
    writeFooter(b, size, false, allocPrev or AllocMiniMask)
  else:
    writeHeader(b, size, false, allocPrev)
    writeFooter(b, size, false, allocPrev)
  # Try to coalesce the block with its neighbors
  b = coalesceBlock(b, mini)
  assert mmCheckheap(instantiationInfo().line)

proc realloc(p: pointer, size: int): pointer =
  ## Redo the dynamic memory allocation
  ## copy the old data to new space
  ## return nil on FAIL or size of zero
  ## Argv: ptr to old payload starting address and realloc size
  ## Pre: ptr is legal pointer to payload starting address
  ## Post: The old payload of size size is in a new allocated space
  if size == 0:
    # If size == 0, then free block and return nil
    free(p)
    return nil
  if p == nil:
    # If p is nil, then equivalent to malloc
    return malloc(size)
  # Otherwise, proceed with reallocation
  let b = payloadToHeader(p)
  var copysize: int
  let blockMini = (b.header and AllocMiniMask) != 0
  if blockMini:
    copysize = WSize
  else:
    copysize = getPayloadSize(b) # gets size of old payload
  let newptr = malloc(size)
  # If malloc fails, the original block is left untouched
  if newptr == nil:
    return nil
  # Copy the old data
  if size < copysize:
    copysize = size
  copyMem(newptr, p, copysize)
  # Free the old block
  free(p)
  return newptr

proc calloc*(elements: int, size: int): pointer =
  ## Allocate elements elements of size bytes each
  ## all initialized to 0 or return nil on FAIL
  ## Argv: Number of elements and size
  ## Pre: Elements and size are non-negative
  ## Post: Heap is allocated and zeroed with the request
  let asize = elements * size
  if asize div elements != size:
    # Multiplication overflowed
    return nil
  result = alloc(asize)
  if result == nil:
    return nil
  # Initialize all bits to 0
  zeroMem(result, asize)
