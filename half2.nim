type
  float16* {.importc: "_Float16", size: sizeof(int16).} = object

proc fromHalf*(x: float16): float32 {.importc: "(NF32)", nodecl.}
proc toHalf*(x: float32): float16 {.importc: "(_Float16)", nodecl.}

proc `+`*(a, b: float16): float16 {.inline.} =
  {.emit: "`result` = `a` + `b`;".}
proc `-`*(a, b: float16): float16 {.inline.} =
  {.emit: "`result` = `a` - `b`;".}
proc `*`*(a, b: float16): float16 {.inline.} =
  {.emit: "`result` = `a` * `b`;".}
proc `/`*(a, b: float16): float16 {.inline.} =
  {.emit: "`result` = `a` / `b`;".}

when isMainModule:
  let a = toHalf(0.125'f32)
  let b = a + toHalf(0.25'f32)
  echo fromHalf(b)
