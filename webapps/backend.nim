import std/[asynchttpserver, asyncdispatch, random], packedjson

type
  Quote = object
    text, author: string

proc main =
  let quotes = @[
    Quote(text: "One thing i know, that i know nothing. This is the source of my wisdom.", author: "Socrates"),
    Quote(text: "Love is composed of a single soul inhabiting two bodies.", author: "Socrates"),
    Quote(text: "There is nothing permanent except change.", author: "Socrates"),
    Quote(text: "I am indebted to my father for living, but to my teacher for living well.", author: "Plutarch"),
    Quote(text: "He who steals a little steals with the same wish as he who steals much, but with less power.", author: "Epicurus"),
    Quote(text: "Let no man be called happy before his death. Till then, he is not happy, only lucky.", author: "Xenophon"),
    Quote(text: "By all means, get married: if you find a good wife, you'll be happy; if not, you'll become a philosopher.", author: "Demosthenes"),
    Quote(text: "Small opportunities are often the beginning of great enterprises.", author: "Pericles")
  ]

  let httpServer = newAsyncHttpServer()
  proc handler(req: Request) {.async.} =
    if req.url.path == "/":
      let headers = {"Content-Type": "text/html"}
      await req.respond(Http200, readFile("app.html"), headers.newHttpHeaders())
    elif req.url.path == "/app.js":
      let headers = {"Content-Type": "application/javascript"}
      await req.respond(Http200, readFile("app.js"), headers.newHttpHeaders())
    elif req.url.path == "/quote":
      let headers = {"Content-type": "application/json"}
      await req.respond(Http200, %quotes.sample, headers.newHttpHeaders())
    else:
      await req.respond(Http404, "")

  waitFor httpServer.serve(Port(8080), handler)

main()
