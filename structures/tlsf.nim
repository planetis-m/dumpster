
const
  MemAlign = 8

  SmallChunkSize = PageSize
  MaxFli = when sizeof(int) > 2: 30 else: 14
  MaxLog2Sli = 5 # 32, this cannot be increased without changing 'uint32'
                 # everywhere!
  MaxSli = 1 shl MaxLog2Sli
  FliOffset = 6
  RealFli = MaxFli - FliOffset

  # size of chunks in last matrix bin
  MaxBigChunkSize = int(1'i32 shl MaxFli - 1'i32 shl (MaxFli-MaxLog2Sli-1))
  HugeChunkSize = MaxBigChunkSize + 1

type
  PChunk = ptr BaseChunk
  PBigChunk = ptr BigChunk
  PSmallChunk = ptr SmallChunk
  BaseChunk {.pure, inheritable.} = object
    prevSize: int        # size of previous chunk; for coalescing
                         # 0th bit == 1 if 'used
    size: int            # if < PageSize it is a small chunk
    owner: ptr MemRegion

  SmallChunk = object of BaseChunk
    next, prev: PSmallChunk  # chunks of the same size
    free: int            # how many bytes remain
    acc: int             # accumulator for small object allocation
    data {.align: MemAlign.}: UncheckedArray[byte]      # start of usable memory

  BigChunk = object of BaseChunk # not necessarily > PageSize!
    next, prev: PBigChunk    # chunks of the same (or bigger) size
    data {.align: MemAlign.}: UncheckedArray[byte]      # start of usable memory

  MemRegion = object
    flBitmap: uint32
    slBitmap: array[RealFli, uint32]
    matrix: array[RealFli, array[MaxSli, PBigChunk]]

template smallChunkOverhead(): untyped = sizeof(SmallChunk)
template bigChunkOverhead(): untyped = sizeof(BigChunk)

proc roundup(x, v: int): int {.inline.} =
  result = (x + (v-1)) and not (v-1)

const
  fsLookupTable: array[byte, int8] = [
    -1'i8, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7
  ]

proc msbit(x: uint32): int {.inline.} =
  let a = if x <= 0xff_ff'u32:
            (if x <= 0xff: 0 else: 8)
          else:
            (if x <= 0xff_ff_ff'u32: 16 else: 24)
  result = int(fsLookupTable[byte(x shr a)]) + a

proc lsbit(x: uint32): int {.inline.} =
  msbit(x and ((not x) + 1))

proc setBit(nr: int; dest: var uint32) {.inline.} =
  dest = dest or (1u32 shl (nr and 0x1f))

proc clearBit(nr: int; dest: var uint32) {.inline.} =
  dest = dest and not (1u32 shl (nr and 0x1f))

proc mappingSearch(r, fl, sl: var int) {.inline.} =
  #let t = (1 shl (msbit(uint32 r) - MaxLog2Sli)) - 1
  # This diverges from the standard TLSF algorithm because we need to ensure
  # PageSize alignment:
  let t = roundup((1 shl (msbit(uint32 r) - MaxLog2Sli)), PageSize) - 1
  r = r + t
  r = r and not t
  r = min(r, MaxBigChunkSize).int
  fl = msbit(uint32 r)
  sl = (r shr (fl - MaxLog2Sli)) - MaxSli
  dec fl, FliOffset
  sysAssert((r and PageMask) == 0, "mappingSearch: still not aligned")

# See http://www.gii.upv.es/tlsf/files/papers/tlsf_desc.pdf for details of
# this algorithm.

proc mappingInsert(r: int): tuple[fl, sl: int] {.inline.} =
  sysAssert((r and PageMask) == 0, "mappingInsert: still not aligned")
  result.fl = msbit(uint32 r)
  result.sl = (r shr (result.fl - MaxLog2Sli)) - MaxSli
  dec result.fl, FliOffset

template mat(): untyped = a.matrix[fl][sl]

proc findSuitableBlock(a: MemRegion; fl, sl: var int): PBigChunk {.inline.} =
  let tmp = a.slBitmap[fl] and (not 0u32 shl sl)
  result = nil
  if tmp != 0:
    sl = lsbit(tmp)
    result = mat()
  else:
    fl = lsbit(a.flBitmap and (not 0u32 shl (fl + 1)))
    if fl > 0:
      sl = lsbit(a.slBitmap[fl])
      result = mat()

template clearBits(sl, fl) =
  clearBit(sl, a.slBitmap[fl])
  if a.slBitmap[fl] == 0u32:
    # do not forget to cascade:
    clearBit(fl, a.flBitmap)

proc removeChunkFromMatrix(a: var MemRegion; b: PBigChunk) =
  let (fl, sl) = mappingInsert(b.size)
  if b.next != nil: b.next.prev = b.prev
  if b.prev != nil: b.prev.next = b.next
  if mat() == b:
    mat() = b.next
    if mat() == nil:
      clearBits(sl, fl)
  b.prev = nil
  b.next = nil

proc removeChunkFromMatrix2(a: var MemRegion; b: PBigChunk; fl, sl: int) =
  mat() = b.next
  if mat() != nil:
    mat().prev = nil
  else:
    clearBits(sl, fl)
  b.prev = nil
  b.next = nil

proc addChunkToMatrix(a: var MemRegion; b: PBigChunk) =
  let (fl, sl) = mappingInsert(b.size)
  b.prev = nil
  b.next = mat()
  if mat() != nil:
    mat().prev = b
  mat() = b
  setBit(sl, a.slBitmap[fl])
  setBit(fl, a.flBitmap)

proc isSmallChunk(c: PChunk): bool {.inline.} =
  result = c.size <= SmallChunkSize-smallChunkOverhead()

proc chunkUnused(c: PChunk): bool {.inline.} =
  result = (c.prevSize and 1) == 0

