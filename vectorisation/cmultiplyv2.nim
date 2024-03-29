import std/macros, fusion/astdsl

type
  Matrix*[M, N: static[int]] = array[M * N, float32]

macro multiplyImpl(M, N, K: static[int]; a, b: typed): untyped =
  proc genSymTable(len: int): NimNode =
    result = newNimNode(nnkBracket)
    for i in 1 .. len:
      result.add genSym(nskLet)

  result = buildAst(stmtList):
    let
      an = genSymTable(M*K)
      bn = genSymTable(K*N)
      cn = genSymTable(M*N)
    for i in 0 ..< an.len:
      newLetStmt(an[i], newTree(nnkBracketExpr, a, newLit(i)))
    for i in 0 ..< bn.len:
      newLetStmt(bn[i], newTree(nnkBracketExpr, b, newLit(i)))
    for j in 0 ..< N:
      for i in 0 ..< M:
        newLetStmt(cn[i * N + j]):
          let args = newNimNode(nnkBracket)
          for k in 0 ..< K:
            args.add infix(an[i * K + k], "*", bn[k * N + j])
          nestList(bindSym"+", args)
    returnStmt(cn)

proc `*`*[M, N, K: static[int]](a: Matrix[M, K], b: Matrix[K, N]): Matrix[M, N] {.inline.} =
  multiplyImpl(M, N, K, a, b)

when isMainModule:
  var
    a: Matrix[4, 3] = [1'f32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    b: Matrix[3, 5] = [1'f32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    c = a * b
  echo c
