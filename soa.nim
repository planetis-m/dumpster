import macros

# type
#    Position = ref object
#       x, y, z: float
#    Entity = object
#       position: Position
#
# proc main =
#    var positions = [
#       Position(x: 1, y: 4, z: 9),
#       Position(x: 1, y: 4, z: 9),
#       Position(x: 1, y: 4, z: 9)]
#    var entity = Entity(position: unown positions[0])
#    echo entity.position.x
#    entity.position = nil
#
# main()


macro with(name: typed, body: untyped): untyped =
  echo name.treeRepr
  echo body.treeRepr
  let firstParam = body.params[1]
  let objDef = name.getImpl()
  echo objDef.treerepr
  newNimNode(nnkEmpty)

macro with2(body: untyped): untyped =
  echo body.treeRepr
  newNimNode(nnkEmpty)

template anonymous() {.pragma.}

type
  Position = object
    x, y, z: float
  Entity = object
    position {.anonymous.}: Position

proc print(entity: Entity) {.with: Entity.} =
  echo entity.x, ", ", entity.y


# wide(1024):
#    type
#       Position = object
#          x, y, z: float

# type
#    PositionWide[N: static int] = object
#       x, y, z: array[N, float]
#
# var allPositions = PositionWide[1024]

# type
#    PositionWide = object
#       x, y, z: seq[float]

# var allPositions = PositionWide(
#    x: newSeq[float](1024),
#    y: newSeq[float](1024),
#    z: newSeq[float](1024))

type
  Position = object
    x, y, z: float

wide Position, 1024

type
  Position = object
    index: int

type
  PositionWide = object
    x: ptr array[1024, float]
    y: ptr array[1024, float]
    z: ptr array[1024, float]

proc initPositionWide(len: int): PositionWide =
  let start = alloc0(len *% 3)
  result = PositionWide(
      x: cast[PositionWide.x](start),
      y: cast[PositionWide.y](cast[ByteAddress](start) +% len),
      z: cast[PositionWide.z](cast[ByteAddress](start) +% len *% 2))

var allPositions = initPositionWide(1024)

proc x(p: Position): float {.inline.} =
  allPositions.x[p.index]

proc `x=`(p: Position, v: float) {.inline.} =
  allPositions.x[p.index] = v

var nextIndex = 0

proc position(x, y, z: float): Position =
  allPositions.x[nextIndex] = x
  allPositions.y[nextIndex] = y
  allPositions.z[nextIndex] = z
  result = Position(index: nextIndex)
  nextIndex.inc

type
  Entity = object
    position: Position

var entity = Entity(position: position(1, 4, 9))
