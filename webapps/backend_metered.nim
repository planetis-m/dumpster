import std/[asynchttpserver, asyncdispatch, asyncnet, random, uri, strutils, times]
from std/json import escapeJson

type
  TokenBucket = object
    capacity, tokens, refillRate: float
    lastRefill: Time

proc refill(tb: var TokenBucket) =
  let now = getTime()
  let elapsedSeconds = now - tb.lastRefill
  let refillAmount = (elapsedSeconds.inMilliseconds.float / 1000) * tb.refillRate
  tb.tokens = min(tb.capacity, tb.tokens + refillAmount)
  tb.lastRefill = now

proc consume(tb: var TokenBucket; tokens: float): bool =
  tb.refill()
  if tokens <= tb.tokens:
    tb.tokens -= tokens
    true
  else:
    false

proc newTokenBucket(capacity, refillRate: float): TokenBucket =
  TokenBucket(capacity: capacity, tokens: capacity, refillRate: refillRate, lastRefill: getTime())

var
  tokenBucket = newTokenBucket(10, 1)

type
  Quote = object
    text, author: string

proc `%`(x: Quote): string =
  result = "{\"text\":" & escapeJson(x.text) & ",\"author\":" & escapeJson(x.author) & "}"

template toQ(a, b): untyped =
  Quote(text: a, author: b)

proc main =
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
      let headers = {"Content-Type": "text/html;charset=utf-8"}
      await req.respond(Http200, readFile("app.html"), headers.newHttpHeaders())
    of "/app.js":
      let headers = {"Content-Type": "application/javascript;charset=utf-8"}
      await req.respond(Http200, readFile("app.js"), headers.newHttpHeaders())
    of "/quote":
      if tokenBucket.consume(1):
        let headers = {"Content-type": "application/json;charset=utf-8"}
        await req.respond(Http200, %sample(quotes), headers.newHttpHeaders())
      else:
        let headers = {"Retry-After": "60"}
        await req.respond(Http429, "", headers.newHttpHeaders())
    else:
      await req.respond(Http404, "")

  waitFor httpServer.serve(Port(8000), handler)

main()
