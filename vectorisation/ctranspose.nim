import std/macros, fusion/astdsl

type
  Matrix[M, N: static[int]] = array[M * N, float32]

macro transposeImpl(M, N: static[int]; x, res: typed): untyped =
  result = buildAst(stmtList):
    for i in 0 ..< M:
      for j in 0 ..< N:
        asgn(bracketExpr(res, intLit(intVal = j * M + i)),
             bracketExpr(x, intLit(intVal = i * N + j)))

proc transpose*[M, N: static[int]](x: Matrix[M, N]): Matrix[N, M] {.inline.} =
  transposeImpl(M, N, x, result)

when isMainModule:
  let
    a: Matrix[2, 3] = [1'f32, 2, 3, 4, 5, 6]
    b = transpose(a)
  echo b
