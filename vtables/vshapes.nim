import math

type
   ShapeVTable = object
      area: proc (p: pointer): float {.nimcall.}
      perimeter: proc (p: pointer): float {.nimcall.}

   Shape = object
      obj: pointer
      vtab: ptr ShapeVTable

   CircleVTable = object
      area: proc (c: CirclePtr): float {.nimcall.}
      perimeter: proc (c: CirclePtr): float {.nimcall.}

   Circle = object
      radius: float
   CirclePtr = ptr Circle

   RectangleVTable = object
      area: proc (r: RectanglePtr): float {.nimcall.}
      perimeter: proc (r: RectanglePtr): float {.nimcall.}

   Rectangle = object
      height, width: float
   RectanglePtr = ptr Rectangle

proc circleArea(c: CirclePtr): float =
   Pi * c.radius * c.radius

proc circlePerimeter(c: CirclePtr): float =
   2 * Pi * c.radius

var theCircleVtab = CircleVTable(
   area: circleArea,
   perimeter: circlePerimeter)

proc initCircle(r: float): Circle =
   result.radius = r

proc newCircle(r: float): CirclePtr =
   result = create(Circle)
   result.radius = r

proc toShape(c: var Circle): Shape =
   result.obj = addr c
   result.vtab = cast[ptr ShapeVTable](addr theCircleVtab)

proc toShape(c: CirclePtr): Shape =
   result.obj = c
   result.vtab = cast[ptr ShapeVTable](addr theCircleVtab)

proc rectangleArea(r: RectanglePtr): float =
   r.height * r.width

proc rectanglePerimeter(r: RectanglePtr): float =
   2 * r.height + 2 * r.width

var theRectangleVtab = RectangleVTable(
   area: rectangleArea,
   perimeter: rectanglePerimeter)

proc initRectangle(h, w: float): Rectangle =
   result.height = h
   result.width = w

proc newRectangle(h, w: float): RectanglePtr =
   result = create(Rectangle)
   result.height = h
   result.width = w

proc toShape(r: var Rectangle): Shape =
   result.obj = addr r
   result.vtab = cast[ptr ShapeVTable](addr theRectangleVtab)

proc toShape(r: RectanglePtr): Shape =
   result.obj = r
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
dealloc(c)
