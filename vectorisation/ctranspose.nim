import std/macros, fusion/astdsl

type
  Matrix[M, N: static[int]] = array[M * N, float32]

macro transposeImpl(x, res: typed): untyped =
  result = buildAst(stmtList):
    let typeSym = getTypeInst(x)
    expectLen typeSym, 3
    let M = typeSym[1].intVal
    let N = typeSym[2].intVal
    for i in 0 ..< M:
      for j in 0 ..< N:
        asgn(bracketExpr(res, intLit(intVal = j * M + i)),
             bracketExpr(x, intLit(intVal = i * N + j)))

proc transpose*[M, N: static[int]](x: Matrix[M, N]): Matrix[N, M] {.inline.} =
  transposeImpl(x, result)

when isMainModule:
  let
    a: Matrix[2, 3] = [1'f32, 2, 3, 4, 5, 6]
    b = transpose(a)
  echo b
