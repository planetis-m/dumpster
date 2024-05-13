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
    pn: [
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    ],
    c: ["red", "purple", "blue", "cyan", "green", "lime", "yellow", "orange"],
    cd: [3, 4, 5, 6],
    lk: [
      "sa23uw", "vc2c7q", "xwz7ja", "tvfkyq", "6hh8xw", "cmppiq", "uphlaw", "i874y3", "voon2h"
    ],
    dm: "lock detected!",
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
