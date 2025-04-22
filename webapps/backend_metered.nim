import std/[asynchttpserver, asyncdispatch, random, uri, monotimes, times]
from std/json import escapeJson

type
  SlidingWindow = object # Approximate
    capacity: int
    currentCount, previousCount: int
    windowSize: int
    currentTime: int64

proc allowRequest(sw: var SlidingWindow): bool =
  let now = getMonoTime().ticks
  # If the current time is outside the window, reset the window
  let elapsedTime = now - sw.currentTime
  if elapsedTime > sw.windowSize:
    sw.previousCount =
      if elapsedTime > sw.windowSize * 2: 0 # Handles long pauses
      else: sw.currentCount # Normal window transition
    sw.currentCount = 0
    sw.currentTime = now # no time-aligned windows
  # Calculate the weighted average of the previous and current counts
  let weight = (sw.windowSize - (now - sw.currentTime)) / sw.windowSize
  let estimatedCount = int(sw.previousCount.float * weight) + sw.currentCount
  # Check if the count exceeds the capacity
  if estimatedCount < sw.capacity:
    # Increment the current count and allow the request
    inc sw.currentCount
    true
  else:
    false

proc newSlidingWindow(capacity: Positive, windowSize: Duration): SlidingWindow =
  SlidingWindow(
    capacity: capacity,
    previousCount: capacity, currentCount: 0,
    windowSize: windowSize.inNanoseconds,
    currentTime: getMonoTime().ticks
  )

type
  Quote = object
    text, author: string

proc `%`(x: Quote): string =
  "{\"text\":" & escapeJson(x.text) & ",\"author\":" & escapeJson(x.author) & "}"

template toQ(a, b): untyped =
  Quote(text: a, author: b)

proc main =
  var
    limiter = newSlidingWindow(2, initDuration(seconds=1))

  let quotes = @[
    toQ("One thing I know, that I know nothing. This is the source of my wisdom.", "Socrates"),
    toQ("Love is composed of a single soul inhabiting two bodies.", "Aristotle"),
    toQ("There is nothing permanent except change.", "Heraclitus"),
    toQ("I am indebted to my father for living, but to my teacher for living well.", "Alexander the Great"),
    toQ("He who steals a little steals with the same wish as he who steals much, but with less power.", "Plato"),
    toQ("Let no man be called happy before his death. Till then, he is not happy, only lucky.", "Solon"),
    toQ("By all means, get married: if you find a good wife, you'll be happy; if not, you'll become a philosopher.", "Socrates"),
    toQ("Small opportunities are often the beginning of great enterprises.", "Demosthenes")
  ]

  let httpServer = newAsyncHttpServer()
  proc handler(req: Request) {.async.} =
    case req.url.path
    of "/":
      let headers = newHttpHeaders({"Content-Type": "text/html;charset=utf-8"})
      await req.respond(Http200, readFile("app.html"), headers)
    of "/app.js":
      let headers = newHttpHeaders({"Content-Type": "application/javascript;charset=utf-8"})
      await req.respond(Http200, readFile("app.js"), headers)
    of "/quote":
      if limiter.allowRequest():
        let headers = newHttpHeaders({"Content-type": "application/json;charset=utf-8"})
        await req.respond(Http200, %sample(quotes), headers)
      else:
        let headers = newHttpHeaders({"Retry-After": "60"})
        await req.respond(Http429, "", headers)
    else:
      await req.respond(Http404, "")

  waitFor httpServer.serve(Port(8000), handler)

main()
