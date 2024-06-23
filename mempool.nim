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

when isMainModule:
  import std/strformat

  type
    Vector2D = object
      x, y: float32

  proc main() =
    const
      PoolSize = 5

    # Create the object pool
    var pool = createFixedPool(sizeof(Vector2D), PoolSize)

    echo "Initial pool state:"
    echo(pool)

    # Allocate some vectors
    var vectors: array[PoolSize, ptr Vector2D]
    for i in 0..<PoolSize:
      vectors[i] = cast[ptr Vector2D](alloc(pool))
      if vectors[i] != nil:
        vectors[i].x = float32(i)
        vectors[i].y = float32(i * i)
        echo &"Allocated vector {i}: ", vectors[i][]

    echo "\nAfter allocating all vectors:"
    echo(pool)

    # Free some vectors
    echo "\nFreeing vectors 1 and 3"
    dealloc(pool, vectors[1])
    dealloc(pool, vectors[3])

    echo "\nAfter freeing two vectors:"
    echo(pool)

    # Allocate new vectors, which should reuse the freed memory
    var newVector1 = cast[ptr Vector2D](alloc(pool))
    var newVector2 = cast[ptr Vector2D](alloc(pool))

    if newVector1 != nil:
      newVector1.x = 10.5
      newVector1.y = 20.7
      echo "\nAllocated new vector 1:"
      echo(newVector1[])

    if newVector2 != nil:
      newVector2.x = -5.2
      newVector2.y = 15.3
      echo "Allocated new vector 2:"
      echo(newVector2[])

    echo "\nFinal pool state:"
    echo(pool)

    # Try to allocate one more vector (should fail)
    var extraVector = cast[ptr Vector2D](alloc(pool))
    if extraVector == nil:
      echo "Failed to allocate extra vector (pool is full)"
    else:
      echo "Unexpectedly allocated extra vector"

  # Run the example
  main()
