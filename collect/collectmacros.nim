import macros

macro collect*(body: untyped): untyped =
  # analyse the body, find the deepest expression 'it' and replace it via
  # 'result.add it'
  let res = genSym(nskVar, "collectResult")
  proc t(n, res: NimNode): NimNode =
    case n.kind
    of nnkStmtList, nnkStmtListExpr, nnkBlockStmt, nnkBlockExpr,
       nnkWhileStmt,
       nnkForStmt, nnkIfExpr, nnkIfStmt, nnkTryStmt, nnkCaseStmt,
       nnkElifBranch, nnkElse, nnkElifExpr:
      result = copyNimTree(n)
      if n.len >= 1:
        result[^1] = t(n[^1], res)
    else:
      template adder(res, it) =
         res.add it
      result = getAst adder(res, n)

  let v = newTree(nnkVarSection,
     newTree(nnkIdentDefs, res, newTree(nnkBracketExpr, bindSym"seq",
     newCall(bindSym"typeof", body)), newEmptyNode()))

  result = newTree(nnkStmtListExpr, v, t(body, res), res)
  echo result.treeRepr


when isMainModule:
   const data = [1, 2, 3, 4, 5, 6]

   block test1:
      let stuff = collect:
         var i = -1
         while i < 4:
            inc i
            for it in data:
               if it < 5 and it > 1:
                  it
      assert stuff == @[2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]

   block test2:
      let keys = @["bird", "word"]
      let values = @[5, 2]
      let stuff = collect:
         let len = min(keys.len, values.len)
         for i in 0 ..< len:
            (keys[i], values[i])
      assert stuff == @[("bird", 5), ("word", 2)]
