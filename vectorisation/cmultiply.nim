import std/macros, fusion/astdsl

type
  Matrix*[M, N: static[int]] = array[M * N, float32]

macro multiplyImpl(M, N, K: static[int]; a, b, res: typed): untyped =
  result = buildAst(stmtList):
    for i in 0 ..< N:
      for j in 0 ..< M:
        asgn:
          bracketExpr(res, intLit(intVal = i * M + j))
          let args = buildAst(bracket):
            for k in 0 ..< K:
              infix(bindSym"*", bracketExpr(a, intLit(intVal = i * M + k)),
                  bracketExpr(b, intLit(intVal = k * K + j)))
          nestList(bindSym"+", args)

proc `*`*[M, N, K: static[int]](a: Matrix[M, K], b: Matrix[K, N]): Matrix[M, N] {.inline.} =
  var c: Matrix[M, N]
  multiplyImpl(M, N, K, a, b, c)
  return c

when isMainModule:
  var
    a: Matrix[4, 3] = [1'f32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    b: Matrix[3, 5] = [1'f32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    c = a * b
  echo c
