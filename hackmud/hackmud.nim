import jsffi/[jstrutils, jdict, jjson]

type
  StdLib = ref object

  Target {.importc.} = ref object
    call: proc (loc: JsonNode): cstring

  Context {.importc.} = ref object
    caller: cstring
    this_script: cstring
    calling_script: cstring

# note: manually add the '#' prefix after minifying
proc getStdLib(): StdLib {.importc: "fs.scripts.lib".}

proc ok(s: StdLib): JsonNode {.importcpp.}
proc notImpl(s: StdLib): JsonNode {.importcpp: "#.not_impl()".}

proc log(s: StdLib; message: cstring) {.importcpp.}
# proc getLog(s: StdLib): JSeq {.importcpp: "#.get_log()".}

proc quine(): cstring {.importc: "fs.scripts.quine".}

proc debug[T](ob: T) {.importc: "D".}

proc find1(query, projection: JsonNode): JsonNode {.importcpp: "db.f(@).first()".}
proc upsert(query, command: JsonNode) {.importc: "db.us".}

proc crackEz21(c: Context; args: JsonNode): JsonNode {.exportc.} =
  ## Usage: script {target: #s.some_user.their_loc}
  # let std = getStdLib()
  let target = cast[Target](args["target"])
  var ret = target.call(JsonNode())
  var success = false
  if ret.contains("EZ_21"):
    let attempts = [cstring"open", "release", "unlock"]
    for a in attempts.items:
      let v = %*{ez_21: a}
      ret = target.call(v)
      if ret.contains("LOCK_UNLOCKED"):
        # std.log("Correct password for ez_21 lock: " & a)
        success = true
        break
  result = %*{
    ok: success,
    msg: ret
  }

proc crackEz40(c: Context; args: JsonNode): JsonNode {.exportc.} =
  ## Usage: script {target: #s.some_user.their_loc}
  # let std = getStdLib()
  let target = cast[Target](args["target"])
  var ret = target.call(JsonNode())
  var success = false
  if ret.contains("EZ_40"):
    var cmd = cstring""
    let attempts = [cstring"open", "release", "unlock"]
    let primes = [
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
      43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    ]
    for a in attempts.items:
      let v = %*{ez_40: a}
      ret = target.call(v)
      if not ret.contains("\"" & a):
        cmd = a
        break
    for p in primes.items:
      let v = %*{ez_40: cmd, ez_prime: p}
      ret = target.call(v)
      if ret.contains("LOCK_UNLOCKED"):
        # std.log("Correct password for ez_40 lock: " & p.toCstr)
        success = true
        break
  result = %*{
    ok: success,
    msg: ret
  }

const
  data = """
{"attempts":["open","release","unlock"],"primes":[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]}"""
{.emit: ["//", data, "//"].} # note: manually re-add the js comment after minifying

proc crackEz40Short(c: Context; args: JsonNode): JsonNode {.exportc.} =
  ## Usage: script {target: #s.some_user.their_loc}
  # let std = getStdLib()
  let target = cast[Target](args["target"])
  var ret = target.call(JsonNode())
  var success = false
  if ret.contains("EZ_40"):
    let consts = parse(quine().split("//")[1]) # note: manually escape '/' after minifying
    var cmd = cstring""
    for a in consts["attempts"].items:
      let v = %*{ez_40: a.getStr}
      ret = target.call(v)
      if not ret.contains("\"" & a.getStr):
        cmd = a.getStr
        break
    for p in consts["primes"].items:
      let v = %*{ez_40: cmd, ez_prime: p.getInt}
      ret = target.call(v)
      if ret.contains("LOCK_UNLOCKED"):
        # std.log("Correct password for ez_40 lock: " & p.toCstr)
        success = true
        break
  result = %*{
    ok: success,
    msg: ret
  }
