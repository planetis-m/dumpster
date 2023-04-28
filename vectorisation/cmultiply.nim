import std/macros, fusion/astdsl

type
  Matrix*[M, N: static[int]] = array[M * N, float32]

macro multiplyImpl(a, b, res: typed): untyped =
  result = buildAst(stmtList):
    let typeSym = getTypeInst(a)
    expectLen typeSym, 3
    let M = typeSym[1].intVal
    let N = typeSym[2].intVal
    for i in 0 ..< N:
      for j in 0 ..< M:
        asgn:
          bracketExpr(res, intLit(intVal = i * M + j))
          let args = buildAst(bracket):
            for k in 0 ..< N:
              infix(bindSym"*", bracketExpr(a, intLit(intVal = i * M + k)),
                  bracketExpr(b, intLit(intVal = k * M + j)))
          nestList(bindSym"+", args)

proc `*`*[M, N: static[int]](a, b: Matrix[M, N]): Matrix[N, M] {.noinline.} =
  var c: Matrix[M, N]
  multiplyImpl(a, b, c)
  return c

when isMainModule:
  var
    a, b: Matrix[4, 4] = [1'f32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    c = a * b
  echo c
