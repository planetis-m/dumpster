# https://observablehq.com/@smrq/hackmud-lock-cracking/2
import jsffi/[jstrutils, jjson], std/jsre

type
  Target {.importc.} = ref object
    call: proc (loc: JsonNode): cstring

  Context {.importc.} = ref object
    caller: cstring
    this_script: cstring
    calling_script: cstring

proc debug[T](ob: T) {.importc: "D".}

proc findFirst(query, projection: JsonNode): JsonNode {.importcpp: "db.f(@).first()".}
proc upsert(query, command: JsonNode) {.importc: "db.us".}

proc ezLocksCracker(c: Context; args: JsonNode): JsonNode {.exportc.} =
  ## Usage: script {target: #s.some_user.their_loc}
  # let std = getStdLib()
  let target = cast[Target](args["target"])
  var success = false
  let consts = findFirst(%*{"_id": "consts"}, JsonNode())
  var args = JsonNode()
  var ret = target.call(args)
  while true:
    if ret.contains("Denied access"):
      let matches = match(ret, newRegExp(r"EZ_\d+"))
      if matches.len > 0:
        for a in consts["a"].items:
          args[matches[0]] = a
          ret = target.call(args)
          if not ret.contains("\"" & a.getStr):
            break
        var field: cstring = ""
        case matches[0]
        of "EZ_35":
          field = "digit"
        of "EZ_40":
          field = "ez_prime"
        for d in consts[field].items:
          args[field] = d
          ret = target.call(args)
          if not ret.contains(d.getInt.toCstr):
            break
    else:
      # std.log("Correct password for ez_21 lock: " & args)
      success = true
      break
  result = %*{
    ok: success,
    msg: ret
  }

proc updateEzConsts(): JsonNode {.exportc.} =
  let consts = %*{"_id": "consts"}
  let update = %*{
    "_id": "consts",
    a: ["unlock", "open", "release"],
    digit: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ez_prime: [
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    ],
  }
  upsert(consts, update)
  result = %*{ok: true}
