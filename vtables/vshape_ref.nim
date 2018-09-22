import math

type
   ShapeVTable = object
      area: proc (p: RootRef): float {.nimcall.}
      perimeter: proc (p: RootRef): float {.nimcall.}

   Shape = object
      obj: RootRef
      vtab: ptr ShapeVTable

   CircleVTable = object
      area: proc (c: Circle): float {.nimcall.}
      perimeter: proc (c: Circle): float {.nimcall.}

   Circle = ref object
      radius: float

   RectangleVTable = object
      area: proc (r: Rectangle): float {.nimcall.}
      perimeter: proc (r: Rectangle): float {.nimcall.}

   Rectangle = ref object
      height, width: float

proc circleArea(c: Circle): float =
   Pi * c.radius * c.radius

proc circlePerimeter(c: Circle): float =
   2 * Pi * c.radius

var theCircleVtab = CircleVTable(
   area: circleArea,
   perimeter: circlePerimeter
)

proc newCircle(r: float): Circle =
   new(result)
   result.radius = r

proc toShape(c: Circle): Shape =
   result.obj = cast[RootRef](c)
   result.vtab = cast[ptr ShapeVTable](addr theCircleVtab)

proc rectangleArea(r: Rectangle): float =
   r.height * r.width

proc rectanglePerimeter(r: Rectangle): float =
   2 * r.height + 2 * r.width

var theRectangleVtab = RectangleVTable(
   area: rectangleArea,
   perimeter: rectanglePerimeter
)

proc newRectangle(h, w: float): Rectangle =
   new(result)
   result.height = h
   result.width = w

proc toShape(r: Rectangle): Shape =
   result.obj = cast[RootRef](r)
   result.vtab = cast[ptr ShapeVTable](addr theRectangleVtab)

proc area(s: Shape): float =
   s.vtab.area(s.obj)

proc perimeter(s: Shape): float =
   s.vtab.perimeter(s.obj)

proc volume(s: Shape, height: float): float =
   s.area() * height

var c = newCircle(2.0)
let s = c.toShape
echo s.perimeter
echo volume(s, 4.2)
