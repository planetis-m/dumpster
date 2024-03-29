include karax / prelude
import karax / [jjson, kajax, i18n, languages], std / dom

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

proc shift[T](x: var seq[T]): T {.importcpp.}

# RateLimiter
var
  queue: seq[proc ()] = @[]
  interval: Interval = nil
  isEmptied = true

proc rateLimit(action: proc (), rate: int) =
  if isEmptied:
    isEmptied = false
    interval = setInterval(proc () =
      if queue.len > 0:
        let call = queue.shift()
        call()
      else:
        isEmptied = true
        clearInterval(interval)
    , rate)
  queue.add(action)

# setCurrentLanguage(Language.elGR)

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
        button(id = "btn", `type` = "button"):
          # DONT run all at once!
          # 1. Ditches calls until the previous completed.
          proc onClick() =
            if not loading:
              loading = true
              # ajaxGet("/quote?lang=" & languageToCode[getCurrentLanguage()],
              #     {cstring"Accept": cstring"application/json"}, onQuote)
              ajaxGet("/quote", {cstring"Accept": cstring"application/json"}, onQuote)
          # 2. Throttles superfluous calls.
          var laziness = false
          proc onClick() =
            if not laziness:
              laziness = true
              ajaxGet("/quote", {cstring"Accept": cstring"application/json"}, onQuote)
              discard setTimeout(proc () = laziness = false, 500)
          # 3. Runs on a fixed interval.
          proc onClick() =
            rateLimit(proc () = ajaxGet("/quote", {cstring"Accept": cstring"application/json"}, onQuote), 500)
          # 4. Debounces a single call at the very end.
          var timeout: Timeout = nil
          proc onClick() =
            if timeout != nil:
              clearTimeout(timeout)
            timeout = setTimeout(proc () = ajaxGet("/quote", {cstring"Accept": cstring"application/json"}, onQuote), 500)

          text "New quote"

setRenderer main
