import math

type
   Shape = tuple
      area: proc (): float
      perimeter: proc (): float

   Circle = ref object
      radius: float

   Rectangle = ref object
      height, width: float

proc area(c: Circle): float =
   Pi * c.radius * c.radius

proc perimeter(c: Circle): float =
   2 * Pi * c.radius

proc newCircle(r: float): Circle =
   result.new()
   result.radius = r

proc toShape(c: Circle): Shape =
   result = (
      area: proc (): float = c.area,
      perimeter: proc (): float = c.perimeter)

proc area(r: Rectangle): float =
   r.height * r.width

proc perimeter(r: Rectangle): float =
   2 * r.height + 2 * r.width

proc newRectangle(h, w: float): Rectangle =
   result.new()
   result.height = h
   result.width = w

proc toShape(r: Rectangle): Shape =
   result = (
      area: proc (): float = r.area,
      perimeter: proc (): float = r.perimeter)

proc volume(s: Shape, height: float): float =
   s.area() * height

var c = newCircle(2.0)
let s = c.toShape
echo s.perimeter()
echo volume(s, 4.2)
