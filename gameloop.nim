# based on: https://dewitters.com/dewitters-gameloop/
# http://lspiroengine.com/?p=378

proc run(game: var Game) =
   const
      ticksPerSec = 25
      skippedTicks = 1_000_000_000 div ticksPerSec # to nanosecs per tick
      maxFramesSkipped = 5 # 20% of ticksPerSec

   var lastTime = getMonoTime().ticks
   while true:
      handleInput(game)
      if not game.isRunning: break

      let now = getMonoTime().ticks
      var framesSkipped = 0
      while now - lastTime >= skippedTicks and framesSkipped < maxFramesSkipped:
         game.update()
         lastTime += skippedTicks
         framesSkipped.inc

      if framesSkipped > 0:
        game.render(float32(now - lastTime) / skippedTicks.float32))
