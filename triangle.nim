import math, cairo

type
   Point = tuple
      x, y: float
   Triangle = tuple
      head, left, right: Point
   Triangles = seq[Triangle]

const
   size = 2000.0
   border = 10.0
   level = 12
   image = "triangles.png"

var
   surface: PSurface
   ctx: PContext

proc triangle(): Triangle =
   let x = size / 2.0 - border
   let y = x * sqrt(3.0)
   let v = (size - y) / 2.0

   ((size / 2.0, v), (border, size - v), (size - border, size - v))

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

proc draw(level: Natural, ts: Triangles): Triangles =
   if level <= 0: return
   let x = (ts[0].head.x - ts[0].left.x) / 2.0
   if x < 1.0: return
   let y = (ts[0].left.y - ts[0].head.y) / 2.0
   var next = newSeq[Triangle]()
   for it in ts:
      let (prevHead, prevLeft, prevRight) = it
      let left  = (prevHead.x - x, prevHead.y + y)
      let right = (prevHead.x + x, prevHead.y + y)
      let tail  = (prevHead.x, prevLeft.y)
      fill(left, right, tail)
      next.add([(prevHead, left, right), (left, prevLeft, tail), (right, tail, prevRight)])

   draw(level - 1, next)

proc main() =
   const intSize = size.int32
   surface = imageSurfaceCreate(TFormat.FORMAT_ARGB32, intSize, intSize)
   ctx = surface.create()
   ctx.rectangle(0, 0, size, size)
   ctx.fill()
   ctx.setLineWidth(0.5)
   let t1 = @[triangle()]
   discard draw(level, t1)
   ctx.destroy()
   discard surface.writeToPng(image)
   surface.destroy()

main()
