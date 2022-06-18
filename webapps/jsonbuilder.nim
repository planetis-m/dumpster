import std/[macros, strutils]

const
  initialSize = 1024

proc escapeJsonUnquoted*(s: string; result: var string) =
  ## Converts a string `s` to its JSON representation without quotes.
  ## Appends to `result`.
  for c in s:
    case c
    of '\L': result.add("\\n")
    of '\b': result.add("\\b")
    of '\f': result.add("\\f")
    of '\t': result.add("\\t")
    of '\v': result.add("\\u000b")
    of '\r': result.add("\\r")
    of '"': result.add("\\\"")
    of '\0'..'\7': result.add("\\u000" & $ord(c))
    of '\14'..'\31': result.add("\\u00" & toHex(ord(c), 2))
    of '\\': result.add("\\\\")
    else: result.add(c)

proc escapeJsonUnquoted*(s: string): string =
  ## Converts a string `s` to its JSON representation without quotes.
  result = newStringOfCap(s.len + s.len shr 3)
  escapeJsonUnquoted(s, result)

proc escapeJson*(s: string; result: var string) =
  ## Converts a string `s` to its JSON representation with quotes.
  ## Appends to `result`.
  result.add("\"")
  escapeJsonUnquoted(s, result)
  result.add("\"")

proc escapeJson*(s: string): string =
  ## Converts a string `s` to its JSON representation with quotes.
  result = newStringOfCap(s.len + s.len shr 3)
  escapeJson(s, result)

proc toJson*(s: string; result: var string) =
  ## Generic constructor for JSON data. Creates a new `JString JsonNode`.
  escapeJson(s, result)

proc toJson*(n: BiggestInt; result: var string) =
  ## Generic constructor for JSON data. Creates a new `JInt JsonNode`.
  result.add $n

proc toJson*(n: float; result: var string) =
  ## Generic constructor for JSON data. Creates a new `JFloat JsonNode`.
  result.add $n

proc toJson*(b: bool; result: var string) =
  ## Generic constructor for JSON data. Creates a new `JBool JsonNode`.
  result.add if b: "true" else: "false"

proc toJson*[T](keyVals: openArray[tuple[key: string, val: T]]; result: var string) =
  ## Generic constructor for JSON data. Creates a new `JObject JsonNode`
  if keyvals.len == 0:
    result.add "[]"
  else:
    var comma = false
    result.add "{"
    for key, val in items(keyVals):
      if comma: result.add ","
      else: comma = true
      escapeJson(key, result)
      result.add ":"
      toJson(val, result)
    result.add "}"

proc toJson*[T](elements: openArray[T]; result: var string) =
  ## Generic constructor for JSON data. Creates a new `JArray JsonNode`
  var comma = false
  result.add "["
  for elem in elements:
    if comma: result.add ","
    else: comma = true
    toJson(elem, result)
  result.add "]"

proc toJson*(o: object; result: var string) =
  ## Generic constructor for JSON data. Creates a new `JObject JsonNode`
  var comma = false
  result.add "{"
  for key, val in o.fieldPairs:
    if comma: result.add ","
    else: comma = true
    escapeJson(key, result)
    result.add ":"
    toJson(val, result)
  result.add "}"

proc toJson*(o: ref object; result: var string) =
  ## Generic constructor for JSON data. Creates a new `JObject JsonNode`
  if o.isNil:
    result.add "null"
  else:
    toJson(o[], result)

proc toJson*(o: enum; result: var string) =
  ## Construct a JsonNode that represents the specified enum value as a
  ## string. Creates a new ``JString JsonNode``.
  toJson($o, result)

proc `%`*[T](v: T): string =
  result = newStringOfCap(initialSize)
  toJson(v, result)
