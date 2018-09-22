import math

type
   Shape = ref object of RootObj
      areaImpl: proc (s: Shape): float {.nimcall.}
      perimeterImpl: proc (s: Shape): float {.nimcall.}

   Circle = ref object of Shape
      radius: float

   Rectangle = ref object of Shape
      height, width: float

proc circleArea(s: Shape): float =
   var s = Circle(s)
   Pi * s.radius * s.radius

proc circlePerimeter(s: Shape): float =
   var s = Circle(s)
   2 * Pi * s.radius

proc newCircle(r: float): Circle =
   new(result)
   result.areaImpl = circleArea
   result.perimeterImpl = circlePerimeter
   result.radius = r

proc rectangleArea(s: Shape): float =
   var s = Rectangle(s)
   s.height * s.width

proc rectanglePerimeter(s: Shape): float =
   var s = Rectangle(s)
   2 * s.height + 2 * s.width

proc newRectangle(h, w: float): Rectangle =
   new(result)
   result.areaImpl = rectangleArea
   result.perimeterImpl = rectanglePerimeter
   result.height = h
   result.width = w

proc area(s: Shape): float =
   s.areaImpl(s)

proc perimeter(s: Shape): float =
   s.perimeterImpl(s)

proc volume(s: Shape, height: float): float =
   s.area() * height

var c = newCircle(2.0)
echo c.perimeter
echo volume(c, 4.2)
