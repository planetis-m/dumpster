type
  float16* {.importc: "_Float16", size: sizeof(int16).} = object

proc fromHalf*(x: float16): float32 {.importc: "(NF32)", nodecl.}
proc toHalf*(x: float32): float16 {.importc: "(_Float16)", nodecl.}

when isMainModule:
  let x = toHalf(0.125'f32)
  echo fromHalf(x)
