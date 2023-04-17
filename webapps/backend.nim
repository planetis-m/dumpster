import std/[asynchttpserver, asyncdispatch, random], packedjson

type
  Quote = object
    text, author: string

proc `%`(x: Quote): string =
  result = "{\"text\":" & escapeJson(x.text) & ",\"author\":" & escapeJson(x.author) & "}"

template toQ(a, b): untyped =
  Quote(text: a, author: b)

proc main =
  let quotes = @[
    toQ("One thing i know, that i know nothing. This is the source of my wisdom.", "Socrates"),
    toQ("Love is composed of a single soul inhabiting two bodies.", "Socrates"),
    toQ("There is nothing permanent except change.", "Socrates"),
    toQ("I am indebted to my father for living, but to my teacher for living well.", "Plutarch"),
    toQ("He who steals a little steals with the same wish as he who steals much, but with less power.", "Epicurus"),
    toQ("Let no man be called happy before his death. Till then, he is not happy, only lucky.", "Xenophon"),
    toQ("By all means, get married: if you find a good wife, you'll be happy; if not, you'll become a philosopher.", "Demosthenes"),
    toQ("Small opportunities are often the beginning of great enterprises.", "Pericles")
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
