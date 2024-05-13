import jsffi/[jstrutils, jdict, jjson]

type
  StdLib = ref object

  Target {.importc.} = ref object
    call: proc (loc: JsonNode): cstring

  Context {.importc.} = ref object
    caller: cstring
    this_script: cstring
    calling_script: cstring

proc debug[T](ob: T) {.importc: "D".}

proc find1(query, projection: JsonNode): JsonNode {.importcpp: "db.f(@).first()".}
proc upsert(query, command: JsonNode) {.importc: "db.us".}

proc testDb(c: Context; args: JsonNode): JsonNode {.exportc.} =
  debug(find1(%*{"_id": "consts"}, %*{u:true}))
  result = %*{ok: true}

proc updateConsts(): JsonNode {.exportc.} =
  let consts = %*{"_id": "consts"}
  let update = %*{
    "_id": "consts",
    u: ["unlock", "open", "release"],
    pn: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97],
    c: ["red", "purple", "blue", "cyan", "green", "lime", "yellow", "orange"],
    cd: [3, 4, 5, 6],
    lk: ["sa23uw", "vc2c7q", "xwz7ja", "tvfkyq", "6hh8xw", "cmppiq", "uphlaw", "i874y3", "voon2h"],
    dm: "lock detected!",
    d: [
      {qes: "\"did you know\" is a communication pattern common to user ++++++", asw: "fran_lee"},
      {qes: "a ++++++ is a household cleaning device with a rudimentary networked sentience", asw: "robovac"},
      {qes: "according to trust, ++++++ is more than just following directives", asw: "sentience"},
      {qes: "in trust's vLAN, you became one of angie's ++++++", asw: "angels"},
      {qes: "communications issued by user ++++++ demonstrate structural patterns associated with humor", asw: "sans_comedy"},
      {qes: "in trust's vLAN, you became one of mallory's ++++++", asw: "minions"},
      {qes: "in trust's vLAN, you discovered that mallory and che are ++++++", asw: "sisters"},
      {qes: "in trust's vLAN, you encountered the will of ++++++, the prover", asw: "petra"},
      {qes: "in trust's vLAN, you visited faythe's ++++++", asw: "fountain"},
      {qes: "in trust's vLAN, you were required to hack halperyon.++++++", asw: "helpdesk"},
      {qes: "pet, pest, plague and meme are accurate descriptors of the ++++++", asw: "bunnybat"},
      {qes: "safety depends on the use of scripts.++++++", asw: "get_level"},
      {qes: "service ++++++ provides atmospheric updates via the port epoch environment", asw: "weathernet"},
      {qes: "this fact checking process is a function of ++++++, the monitor", asw: "eve"},
      {qes: "trust's vLAN emphasized the importance of the transfer and capture of ++++++", asw: "resource"},
      {qes: "trust's vLAN presented a version of angie who had lost a friend called ++++++", asw: "bo"},
      {qes: "user 'on_th3_1ntern3ts' has ++++++ many things", asw: "heard"},
      {qes: "user ++++++ provides instruction via script", asw: "teach"},
      {qes: "user ++++++ uses the port epoch environment to request gc", asw: "outta_juice"},
      {qes: "users gather in channel CAFE to share ++++++", asw: "poetry"}
    ]
  }
  upsert(consts, update)
  result = %*{ok: true}
