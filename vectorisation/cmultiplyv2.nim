import std/macros, fusion/astdsl

type
  Matrix*[M, N: static[int]] = array[M * N, float32]

macro multiplyImpl(M, N, K: static[int]; a, b, res: typed): untyped =
  result = buildAst(stmtList):
    for i in 0 ..< M*K:
      newLetStmt(ident("a" & $i), newTree(nnkBracketExpr, a, newLit(i)))
    for i in 0 ..< K*N:
      newLetStmt(ident("b" & $i), newTree(nnkBracketExpr, b, newLit(i)))
    for j in 0 ..< N:
      for i in 0 ..< M:
        newLetStmt(ident("c" & $(i*N+j))):
          let args = newNimNode(nnkBracket)
          for k in 0 ..< K:
            args.add infix(ident("a" & $(i*K+k)), "*", ident("b" & $(k*N+j)))
          nestList(bindSym"+", args)
    returnStmt:
      bracket:
        for i in 0 ..< M*N:
          ident("c" & $i)

proc `*`*[M, N, K: static[int]](a: Matrix[M, K], b: Matrix[K, N]): Matrix[M, N] {.inline.} =
  multiplyImpl(M, N, K, a, b, result)

when isMainModule:
  var
    a: Matrix[4, 3] = [1'f32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    b: Matrix[3, 5] = [1'f32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    c = a * b
  echo c
