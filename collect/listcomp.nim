import macros, sets, tables

proc transLastStmt(n, res, callConstr: NimNode): (NimNode, NimNode, NimNode) =
   # Looks for the last statement of the last statement, etc...
   case n.kind
   of nnkStmtList, nnkStmtListExpr, nnkBlockStmt, nnkBlockExpr, nnkWhileStmt,
         nnkForStmt, nnkIfExpr, nnkIfStmt, nnkTryStmt, nnkCaseStmt,
         nnkElifBranch, nnkElse, nnkElifExpr:
      result[0] = copyNimTree(n)
      result[1] = copyNimTree(n)
      result[2] = copyNimTree(n)
      if n.len >= 1:
         (result[0][^1], result[1][^1], result[2][^1]) = transLastStmt(n[^1],
               res, callConstr)
   of nnkTableConstr:
      result[1] = n[0][0]
      result[2] = n[0][1]
      callConstr.add(bindSym"initTable", newCall(bindSym"typeof", newEmptyNode()),
            newCall(bindSym"typeof", newEmptyNode()))
      template adder(res, k, v) = res[k] = v
      result[0] = getAst(adder(res, n[0][0], n[0][1]))
   of nnkCurly:
      result[2] = n[0]
      callConstr.add(bindSym"initHashSet", newCall(bindSym"typeof",
            newEmptyNode()))
      template adder(res, v) = res.incl(v)
      result[0] = getAst(adder(res, n[0]))
   else:
      result[2] = n
      callConstr.add(bindSym"newSeq", newCall(bindSym"typeof", newEmptyNode()))
      template adder(res, v) = res.add(v)
      result[0] = getAst(adder(res, n))

macro collect*(body): untyped =
   # analyse the body, find the deepest expression 'it' and replace it via
   # 'result.add it'
   let res = genSym(nskVar, "collectResult")
   let callConstr = newNimNode(nnkBracketExpr)
   var (resBody, keyType, valueType) = transLastStmt(body, res, callConstr)
   if callConstr.len == 3:
      callConstr[1][1] = keyType
      callConstr[2][1] = valueType
   else:
      callConstr[1][1] = valueType
   if resBody.kind != nnkBlockStmt:
      resBody = newBlockStmt(resBody)
   result = newTree(nnkStmtListExpr, newVarStmt(res, newTree(nnkCall,
         callConstr)), resBody, res)
   echo result.treeRepr

when isMainModule:
   var data = @["bird", "word"] # if this gets stuck in your head, its not my fault
   assert collect(for (i, d) in data.pairs: (if i mod 2 == 0: d)) == @["bird"]
   assert collect(for (i, d) in data.pairs: {i: d}) == {1: "word",
         0: "bird"}.toTable
   assert collect(for d in data.items: {d}) == data.toHashSet

   let y = collect:
      var data = @["bird", "word"]
      for (i, d) in data.pairs:
         if i mod 2 == 0: d
   echo y
