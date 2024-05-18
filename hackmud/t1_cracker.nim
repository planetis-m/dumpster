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
  let consts = findFirst(%*{"_id": "ez"}, JsonNode())
  var attempts = 1
  var args = JsonNode()
  var ret = target.call(args)
  while attempts <= 5:
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
      inc attempts
    else:
      # std.log("Correct password for ez_21 lock: " & args)
      success = true
      break
  result = %*{
    ok: success,
    msg: ret
  }

proc updateEzConsts(): JsonNode {.exportc.} =
  let consts = %*{"_id": "ez"}
  let update = %*{
    "_id": "ez",
    a: ["unlock", "open", "release"],
    digit: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ez_prime: [
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    ],
  }
  upsert(consts, update)
  result = %*{ok: true}

proc updateC001Consts(): JsonNode {.exportc.} =
  let consts = %*{"_id": "c0"}
  let update = %*{
    "_id": "c0",
    c001: ["red", "purple", "blue", "cyan", "green", "lime", "yellow", "orange"],
    color_digit: [3, 6, 4, 4, 5, 4, 6, 6],
    # c002:
    # c002_complement:
  }
  upsert(consts, update)
  result = %*{ok: true}

proc updateDataCheckConsts(): JsonNode {.exportc.} =
  let consts = %*{"_id": "dcheck"}
  let update = %*{
    "_id": "dcheck",
    d: [
      ["\"did you know\" is a communication pattern common to user ++++++", "fran_lee"],
      ["a ++++++ is a household cleaning device with a rudimentary networked sentience", "robovac"],
      ["according to trust, ++++++ is more than just following directives", "sentience"],
      ["in trust's vLAN, you became one of angie's ++++++", "angels"],
      ["communications issued by user ++++++ demonstrate structural patterns associated with humor", "sans_comedy"],
      ["in trust's vLAN, you became one of mallory's ++++++", "minions"],
      ["in trust's vLAN, you discovered that mallory and che are ++++++", "sisters"],
      ["in trust's vLAN, you encountered the will of ++++++, the prover", "petra"],
      ["in trust's vLAN, you visited faythe's ++++++", "fountain"],
      ["in trust's vLAN, you were required to hack halperyon.++++++", "helpdesk"],
      ["pet, pest, plague and meme are accurate descriptors of the ++++++", "bunnybat"],
      ["safety depends on the use of scripts.++++++", "get_level"],
      ["service ++++++ provides atmospheric updates via the port epoch environment", "weathernet"],
      ["this fact checking process is a function of ++++++, the monitor", "eve"],
      ["trust's vLAN emphasized the importance of the transfer and capture of ++++++", "resource"],
      ["trust's vLAN presented a version of angie who had lost a friend called ++++++", "bo"],
      ["user 'on_th3_1ntern3ts' has ++++++ many things", "heard"],
      ["user ++++++ provides instruction via script", "teach"],
      ["user ++++++ uses the port epoch environment to request gc", "outta_juice"],
      ["users gather in channel CAFE to share ++++++", "poetry"]
    ]
  }
  upsert(consts, update)
  result = %*{ok: true}
