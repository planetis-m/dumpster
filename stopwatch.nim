import std / [times, monotimes]

type
  StopwatchState = enum
    Waiting, ## Initial state with an elapsed time value of 0 seconds.
    Started, ## Stopwatch has started counting the elapsed time since this `Instant`
             ## and accumuluated time from previous start/stop cycles `Duration`.
    Ended,   ## Stopwatch has been stopped and reports the elapsed time `Duration`.

  Stopwatch* = object ## A stopwatch which accurately measures elapsed time.
    case state: StopwatchState
    of Waiting: ## Initial state with an elapsed time value of 0 seconds.
      discard
    of Started: ## Stopwatch has started counting the elapsed time since this `Monotime`
                ## and accumuluated time from previous start/stop cycles `Duration`.
      dur1: Duration
      start: Monotime
    of Ended: ## Stopwatch has been stopped and reports the elapsed time `Duration`.
      dur2: Duration

proc initStopwatch*(): Stopwatch =
  ## Creates a new stopwatch.
  result = Stopwatch(state: Waiting)

proc elapsed(self: Monotime): Duration =
  result = getMonoTime() - self

proc elapsed*(self: Stopwatch): Duration =
  ## Retrieves the elapsed time.
  case self.state
  of Waiting: result = initDuration(0, 0)
  of Started: result = self.dur1 + self.start.elapsed()
  of Ended: result = self.dur2

proc restart*(self: var Stopwatch) =
  ## Stops, resets, and starts the stopwatch again.
  self = Stopwatch(state: Started, dur1: initDuration(0, 0), start: getMonoTime())

proc start*(self: var Stopwatch) =
  ## Starts, or resumes, measuring elapsed time. If the stopwatch has been
  ## started and stopped before, the new results are compounded onto the
  ## existing elapsed time value.
  ##
  ## Note: Starting an already running stopwatch will do nothing.
  case self.state
  of Waiting: self.restart()
  of Ended:
    self = Stopwatch(state: Started, dur1: self.dur2, start: getMonoTime())
  else: discard

proc stop*(self: var Stopwatch) =
  ## Stops measuring elapsed time.
  ##
  ## Note: Stopping a stopwatch that isn't running will do nothing.
  if self.state == Started:
    self = Stopwatch(state: Ended, dur2: self.dur1 + self.start.elapsed())

proc reset*(self: var Stopwatch) =
  ## Clears the current elapsed time value.
  self = Stopwatch(state: Waiting)

when isMainModule:
  var st = initStopwatch()
  start(st)
  echo st.elapsed
  stop(st)
  echo st.elapsed
