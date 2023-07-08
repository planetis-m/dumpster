import std/[asynchttpserver, asyncdispatch, algorithm, random, uri]
import karax/languages
from std/json import escapeJson

type
  Quote = object
    text, author: string

proc `%`(x: Quote): string =
  result = "{\"text\":" & escapeJson(x.text) & ",\"author\":" & escapeJson(x.author) & "}"

template toQ(a, b): untyped =
  Quote(text: a, author: b)

const
  defaultLanguage = Language.enUS

proc parseLanguage(s: string): Language =
  let i = binarySearch(languageToCode, s)
  if i >= 0: result = Language(i)
  else: result = defaultLanguage

proc getLanguage(url: Uri): Language =
  var lang = ""
  for key, val in decodeQuery(url.query):
    if key == "lang":
      lang = val
      break
  result = parseLanguage(lang)

proc randQuote(lang: Language): Quote =
  case lang
  of Language.enUS:
    result = sample(@[
      toQ("One thing I know, that I know nothing. This is the source of my wisdom.", "Socrates"),
      toQ("Love is composed of a single soul inhabiting two bodies.", "Aristotle"),
      toQ("There is nothing permanent except change.", "Heraclitus"),
      toQ("I am indebted to my father for living, but to my teacher for living well.", "Alexander the Great"),
      toQ("He who steals a little steals with the same wish as he who steals much, but with less power.", "Plato"),
      toQ("Let no man be called happy before his death. Till then, he is not happy, only lucky.", "Solon"),
      toQ("By all means, get married: if you find a good wife, you'll be happy; if not, you'll become a philosopher.", "Socrates"),
      toQ("Small opportunities are often the beginning of great enterprises.", "Demosthenes")
    ])
  of Language.elGR:
    result = sample(@[
      toQ("Ένα πράγμα ξέρω, ότι δεν ξέρω τίποτα. Αυτή είναι η πηγή της σοφίας μου.", "Σωκράτης"),
      toQ("Η αγάπη αποτελείται από μια ψυχή που κατοικεί σε δύο σώματα.", "Αριστοτέλης"),
      toQ("Δεν υπάρχει τίποτα μόνιμο εκτός από την αλλαγή.", "Ηράκλειτος"),
      toQ("Στον πατέρα μου οφείλω το ζείν, αλλά στον δάσκαλο μου το ευ ζήν.", "Μέγας Αλέξανδρος"),
      toQ("Αυτός που κλέβει λίγα κλέβει με την ίδια επιθυμία με αυτόν που κλέβει πολλά, αλλά με μικρότερη δύναμη.", "Πλάτων"),
      toQ("Μη μακαρίζεις κανένα πριν δεις το τέλος του.", "Σόλων"),
      toQ("Αν βρεις μια καλή σύζυγο, θα είσαι ευτυχής· αν όχι, θα γίνεις φιλόσοφος.", "Σωκράτης"),
      toQ("Οι μικρές ευκαιρίες συχνά είναι η αρχή των μεγάλων επιχειρήσεων.", "Δημοσθένης")
    ])
  else: discard

proc main =
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
      let headers = {"Content-type": "application/json;charset=utf-8"}
      await req.respond(Http200, %randQuote(getLanguage(req.url)), headers.newHttpHeaders())
    else:
      await req.respond(Http404, "")

  waitFor httpServer.serve(Port(8000), handler)

main()
