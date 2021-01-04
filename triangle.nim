import math, cairo

type
   Point = object
      x, y: float
   Triangle = object
      head, left, right: Point

const
   size = 2000.0
   border = 10.0
   level = 12
   image = "triangles.png"

proc triangle(): Triangle =
   let x = size / 2.0 - border
   let y = x * sqrt(3.0)
   let v = (size - y) / 2.0
   result = Triangle(
      head: Point(x: size / 2.0, y: v),
      left: Point(x: border, y: size - v),
      right: Point(x: size - border, y: size - v))

proc fill(left, right, tail: Point) =
   let
      r = (size - tail.y) / size
      g = (size - tail.x) / size
      b = tail.x / size
   ctx.setSourceRgb(r, g, b)
   ctx.moveTo(left.x, left.y)
   ctx.lineTo(right.x, right.y)
   ctx.lineTo(tail.x, tail.y)
   ctx.fill()

proc draw(level: Natural, ts: seq[Triangle]) =
   if level > 0:
      let x = (ts[0].head.x - ts[0].left.x) / 2.0
      if x >= 1.0:
         let y = (ts[0].left.y - ts[0].head.y) / 2.0
         var next: seq[Triangle]
         for prev in ts.items:
            let left = Point(x: prev.head.x - x, y: prev.head.y + y)
            let right = Point(x: prev.head.x + x, y: prev.head.y + y)
            let tail = Point(x: prev.head.x, y: prev.left.y)
            fill(left, right, tail)
            next.add([
               Triangle(head: prev.head, left: left, right: right),
               Triangle(head: left, left: prev.left, right: tail),
               Triangle(head: right, left: tail, right: prev.right)])
         draw(level - 1, next)

proc main() =
   const intSize = size.int32
   let surface = imageSurfaceCreate(TFormat.FORMAT_ARGB32, intSize, intSize)
   let ctx = surface.create()
   ctx.rectangle(0, 0, size, size)
   ctx.fill()
   ctx.setLineWidth(0.5)
   let t1 = @[triangle()]
   draw(level, t1)
   ctx.destroy()
   checkStatus surface.writeToPng(image)
   surface.destroy()

main()
