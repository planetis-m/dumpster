# based on: https://dewitters.com/dewitters-gameloop/
# http://lspiroengine.com/?p=378

import std/monotimes

proc run(game: var Game) =
  const
    TickRate = 25
    TickDuration = 1_000_000_000 div TickRate # to nanosecs per tick
    MaxTicks = 5 # 20% of tickRate

  var
    lastTime = getMonoTime().ticks
    accumulator = 0'i64

  while true:
    handleEvents(game)
    if not game.isRunning: break

    let now = getMonoTime().ticks
    accumulator += now - lastTime
    lastTime = now

    var ticks = 0
    while accumulator >= TickDuration and ticks < MaxTicks:
      game.update()
      accumulator -= TickDuration
      inc ticks

    if ticks > 0:
      let alpha = accumulator.float32 / TickDuration / 1_000_000_000
      game.render(alpha)
