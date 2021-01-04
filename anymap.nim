import strutils, macros, macrocache

# type
#    Position = object
#       x, y: float
#
#    Velocity = object
#       x, y: float
#
#    Acceleration = object
#       x, y: float
#
#    AnyMap = object
#       data0: seq[Position]
#       data1: seq[Velocity]
#       data2: seq[Acceleration]
#
# proc `[]`(a: AnyMap, t: typedesc[Position]): seq[Position] =
#    result = a.data0
#
# proc `[]`(a: var AnyMap, t: typedesc[Position]): var seq[Position] =
#    result = a.data0
#
# proc `[]`(a: AnyMap, t: typedesc[Velocity]): seq[Velocity] =
#    result = a.data1
#
# proc `[]`(a: var AnyMap, t: typedesc[Velocity]): var seq[Velocity] =
#    result = a.data1
#
# proc main =
#    var a: AnyMap
#    let ent1 = 0'u16
#
#    a[Position].add Position(x: 0, y: 0)
#    a[Velocity].add Velocity(x: 1, y: 0)
#
#    a[Position][ent1].x += a[Velocity][ent1].x
#    a[Position][ent1].y += a[Velocity][ent1].y
#
#    echo a[Position][ent1].x
#
# main()

const componentRegistry = CacheSeq"anymap.componentRegistry"

proc makeField(n: NimNode): NimNode =
   let s = n.strVal
   result = ident(toLowerAscii(s[0]) & substr(s, 1))

macro component*(s: untyped): untyped =
   expectKind s, nnkTypeDef
   result = copyNimTree(s)
   componentRegistry.add result[0].basename

macro world*(s: untyped): untyped =
   expectKind s, nnkTypeDef
   result = copyNimTree(s)
   if result[2][2].kind == nnkEmpty:
      result[2][2] = newNimNode(nnkRecList)

   template makeStorage(component): untyped =
      seq[component]

   for c in items(componentRegistry):
      result[2][2].add newIdentDefs(c.makeField, getAst(makeStorage(c)))

when isMainModule:
   type
      Position {.component.} = object
         x, y: float

      Velocity {.component.} = object
         x, y: float

      Acceleration {.component.} = object
         x, y: float

      World {.world.} = object
         signatures: seq[uint16]

