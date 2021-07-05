include karax / prelude
import karax / [jjson, kajax], std / dom

type
  Quote = ref object
    text, author: cstring

var
  current = Quote(text: "", author: "")
  status = 0
  loading = false

proc onQuote(httpStatus: int, response: cstring) =
  loading = false
  status = httpStatus
  if httpStatus == 200:
    current = fromJson[Quote](response)

# RateLimiter
var
  queue: seq[proc ()] = @[]
  timeOutRef: Interval = nil
  wasEmptied = true

proc rateLimit(action: proc (), rate: int) =
  if wasEmptied:
    wasEmptied = false
    timeOutRef = setInterval(proc () =
      if queue.len > 0:
        let call = queue.pop()
        call()
      else:
        wasEmptied = true
        clearInterval(timeOutRef)
    , rate)
  queue.add(action)

proc main(): VNode =
  result = buildHtml(tdiv):
    tdiv(class = "content-box"):
      tdiv(class = "logo"):
        text "nim"
      tdiv(class = "quote-box"):
        h1(class = "quote"):
          text "Random Quote Generator"
        if loading:
          p(class = "quoteLoading"):
            text "loading"
        elif status == 404:
          p(class = "error404"):
            text "404 Not Found"
        elif status == 200:
          h4(class = "quoteDisplay"):
            text current.text
          p(id = "author"):
            text current.author
        button(id = "btn"):
          # Ditches calls until the previous completed
          proc onClick() =
            if not loading:
              loading = true
              ajaxGet("/quote", {cstring"Accept": cstring"application/json"}, onQuote)
          # Ditches superfluous calls
          var laziness = false
          proc onClick() =
            if not laziness:
              laziness = true
              ajaxGet("/quote", {cstring"Accept": cstring"application/json"}, onQuote)
              discard setTimeout(proc () = laziness = false, 500)
          # Runs on a fixed interval
          proc onClick() =
            rateLimit(proc () = ajaxGet("/quote", {cstring"Accept": cstring"application/json"}, onQuote), 500)

          text "New quote"

setRenderer main
