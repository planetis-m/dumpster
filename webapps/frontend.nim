include karax / prelude
import karax / [jjson, kajax]

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
          proc onClick() =
            if not loading:
              loading = true
              ajaxGet("/quote", {cstring"Accept": cstring"application/json"}, onQuote)
          text "New quote"

setRenderer main
