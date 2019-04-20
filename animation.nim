import times, cairo

const
   width = 100
   height = 60

var rocket: PSurface

proc animate(scene: func (time: float; ctx: PContext)) =
   # Given a function creates a scene
   const aSixth = 1 / 6 # 6 times per second
   var
      timer = 0.0
      past = present
      present = cpuTime()

      surface = imageSurfaceCreate(FormatArgb32, width, height)
      ctx = surface.create()

   while true:
      past = present
      present = cpuTime()
      let tick = present - past
      timer += tick
      if timer >= aSixth:
         scene(present, ctx)
         discard surface.writeToPng("rocketlanding.png")
         timer -= aSixth

   ctx.destroy()
   surface.destroy()

proc travelDistance(t: float): float =
   # Compute the distance traveled by the rocket in a time
   v * t

proc pictureOfRocket(time: float; ctx: PContext) =
   # Place rocket in a scene with height
   const
      x = 50
      v = 3
   let centerToTop = height - rocket.getHeight.float / 2.0
   let d = travelDistance(time)
   if d <= centerToTop:
      ctx.setSource(rocket, x, d)
   else:
      ctx.setSource(rocket, x, centerToTop)

rocket = imageSurfaceCreateFromPng("data/rocket.png")
animate(pictureOfRocket)
rocket.destroy()
