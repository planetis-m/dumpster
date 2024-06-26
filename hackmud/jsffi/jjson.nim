## This module implements some small zero-overhead 'JsonNode' type
## and helper that maps directly to JavaScript objects.

type
  JsonNode* {.importc.} = ref object

proc `[]`*(obj: JsonNode; fieldname: cstring): JsonNode {.importcpp: "#[#]".}
proc `[]`*(obj: JsonNode; index: int): JsonNode {.importcpp: "#[#]".}
proc `[]=`*[T](obj: JsonNode; fieldname: cstring; value: T)
  {.importcpp: "#[#] = #".}
proc length(x: JsonNode): int {.importcpp: "#.length".}
proc len*(x: JsonNode): int = (if x.isNil: 0 else: x.length)

proc parse*(input: cstring): JsonNode {.importcpp: "JSON.parse(#)".}
proc hasField*(obj: JsonNode; fieldname: cstring): bool {.importcpp: "#[#] !== undefined".}

proc newJArray*(elements: varargs[JsonNode]): JsonNode {.importcpp: "#".}

template `%`*(x: typed): JsonNode = cast[JsonNode](x)
template `%`*(x: string): JsonNode = cast[JsonNode](cstring x)

proc getNum*(x: JsonNode): int {.importcpp: "#".}

proc getInt*(x: JsonNode): int {.importcpp: "#".}
proc getStr*(x: JsonNode): cstring {.importcpp: "#".}
proc getFNum*(x: JsonNode): cstring {.importcpp: "#".}

# iterator items*(x: JsonNode): JsonNode =
#   for i in 0..<len(x): yield x[i]

iterator items*(x: JsonNode): JsonNode =
  var i: int
  {.emit: ["for (", i, " in 0; ", i, " < (", x, " ? ", x, ".length : 0"; ", i"++)) {"].}
  yield x[i]
  {.emit: ["}"].}

# iterator keys*(x: JsonNode): JsonNode =
#   var kkk: JsonNode
#   {.emit: ["for (", kkk, " in ", x, ") {if (", x, ".hasOwnProperty(", kkk,")) {"].}
#   yield kkk
#   {.emit: ["}}"].}

import macros

proc toJson(x: NimNode): NimNode =
  case x.kind
  of nnkBracket:
    result = newCall(bindSym"newJArray")
    for i in 0 ..< x.len:
      result.add(toJson(x[i]))
  of nnkTableConstr:
    let obj = genSym(nskVar, "obj")
    result = newTree(nnkStmtListExpr, newVarStmt(obj, newTree(nnkObjConstr, bindSym"JsonNode")))
    for i in 0 ..< x.len:
      x[i].expectKind nnkExprColonExpr
      let key = x[i][0]
      let a = if key.kind in {nnkIdent, nnkSym, nnkAccQuoted}:
                newLit($key)
              else:
                key
      result.add newAssignment(newTree(nnkBracketExpr, obj, newCall(bindSym"cstring", a)), toJson(x[i][1]))
    result.add obj
  of nnkCurly:
    x.expectLen(0)
    result = newTree(nnkObjConstr, bindSym"JsonNode")
  of nnkNilLit:
    result = newNilLit()
  else:
    result = newCall(bindSym"%", x)

macro `%*`*(x: untyped): untyped =
  ## Convert an expression to a JsonNode directly, without having to specify
  ## `%` for every element.
  result = toJson(x)
