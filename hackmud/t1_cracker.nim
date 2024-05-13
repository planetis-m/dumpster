import jsffi/[jstrutils, jdict, jjson], std/jsre

type
  StdLib = ref object

  Target {.importc.} = ref object
    call: proc (loc: JDict[cstring, JsonNode]): cstring

  Context {.importc.} = ref object
    caller: cstring
    this_script: cstring
    calling_script: cstring

proc debug[T](ob: T) {.importc: "D".}

proc find1(query, projection: JsonNode): JsonNode {.importcpp: "db.f(@).first()".}
proc upsert(query, command: JsonNode) {.importc: "db.us".}

proc ezLocksCracker(c: Context; args: JsonNode): JsonNode {.exportc.} =
  ## Usage: script {target: #s.some_user.their_loc}
  # let std = getStdLib()
  let target = cast[Target](args["target"])
  var success = false
  let consts = find1(%*{"_id": "consts"}, JsonNode())
  var attempts = 1
  var args = newJDict[cstring, JsonNode]()
  var ret = target.call(args)
  while ret.startsWith("Denied access") and attempts <= 5:
    block outer:
      let matches = match(ret, newRegExp(r"EZ_\d+"))
      if matches.len != 0:
        for a in consts["a"].items:
          args[matches[0]] = a
          ret = target.call(args)
          if not ret.contains("\"" & a.getStr):
            break outer
      if ret.contains("EZ_35"):
        for d in 0..9:
          args["digit"] = %d
          ret = target.call(args)
          if not ret.contains(d.toCstr):
            break outer
      if ret.contains("EZ_40"):
        for p in consts["pn"].items:
          args["ez_prime"] = p
          ret = target.call(args)
          if ret.contains(p.getInt.toCstr):
            break outer
      return %*{ok: false}
    if not ret.contains("LOCK_ERROR"):
      # std.log("Correct password for ez_21 lock: " & args)
      success = true
      break
    inc attempts
  result = %*{
    ok: success,
    msg: ret
  }

proc updateConsts(): JsonNode {.exportc.} =
  let consts = %*{"_id": "consts"}
  let update = %*{
    "_id": "consts",
    a: ["unlock", "open", "release"],
    pn: [
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    ],
    c: ["red", "purple", "blue", "cyan", "green", "lime", "yellow", "orange"],
    cd: [3, 4, 5, 6],
    lk: [
      "sa23uw", "vc2c7q", "xwz7ja", "tvfkyq", "6hh8xw", "cmppiq", "uphlaw", "i874y3", "voon2h"
    ],
    d: [
      "fran_lee", "robovac", "sentience", "angels",
      "sans_comedy", "minions", "sisters", "petra",
      "fountain", "helpdesk", "bunnybat", "get_level",
      "weathernet", "eve", "resource", "bo",
      "heard", "teach", "outta_juice", "poetry"
    ]
  }
  upsert(consts, update)
  result = %*{ok: true}
