import macros, sets, tables

proc iterInterval(iterCall: NimNode): NimNode =
   template checkCompiles(expr) =
      when compiles(expr): expr else: 1
   template calcIterations(a, b) =
      b - a + 1
   template calcIterations(a, b, s) =
      (b - a + 1) div s
   # XXX Fix for a.countup(b)
   let op = if iterCall.kind == nnkDotExpr: $iterCall[1] else: $iterCall[0]
   case op
   of "..":
      let lower = iterCall[1]
      let upper = iterCall[2]
      result = getAst(checkCompiles(getAst(calcIterations(lower, upper))))
   of "countup":
      let lower = iterCall[1]
      let upper = iterCall[2]
      if iterCall.len == 3:
         result = getAst(checkCompiles(getAst(calcIterations(lower, upper))))
      else:
         let step = iterCall[3]
         result = getAst(checkCompiles(getAst(calcIterations(lower, upper, step))))
   of "countdown":
      let lower = iterCall[2]
      let upper = iterCall[1]
      if iterCall.len == 3:
         result = getAst(checkCompiles(getAst(calcIterations(lower, upper))))
      else:
         let step = iterCall[3]
         result = getAst(checkCompiles(getAst(calcIterations(lower, upper, step))))
   of "..<":
      let lower = iterCall[1]
      let upper = iterCall[2]
      result = getAst(checkCompiles(newTree(nnkInfix, bindSym"-", upper, lower)))
   of "items", "pairs":
      result = getAst(checkCompiles(newCall(bindSym"len", if iterCall.kind ==
            nnkDotExpr: iterCall[0] else: iterCall[1])))
   of "enumerate":
      result = iterInterval(if iterCall.kind == nnkDotExpr: iterCall[0]
            else: iterCall[1])
   else:
      result = newIntLitNode(1)

proc transLastStmt(n, res, bracketExpr: NimNode, hasIfs: var bool): (
      NimNode, NimNode, NimNode, NimNode) =
   # Looks for the last statement of the last statement, etc...
   case n.kind
   of nnkStmtList, nnkStmtListExpr, nnkBlockStmt, nnkBlockExpr, nnkWhileStmt,
         nnkForStmt, nnkIfExpr, nnkIfStmt, nnkTryStmt, nnkCaseStmt,
         nnkElifBranch, nnkElse, nnkElifExpr:
      result[0] = copyNimTree(n)
      result[1] = copyNimTree(n)
      result[2] = copyNimTree(n)
      hasIfs = hasIfs or n.kind in {nnkIfStmt, nnkIfExpr, nnkCaseStmt}
      if n.len >= 1:
         (result[0][^1], result[1][^1], result[2][^1], result[3][^1]) = transLastStmt(n[^1],
               res, bracketExpr, hasIfs)
      if n.kind == nnkForStmt:
         if not hasIfs:
            result[3] = newTree(nnkInfix, bindSym"*", iterInterval(n[^2]), newEmptyNode())
         else:
            result[3] = newEmptyNode()
   of nnkTableConstr:
      result[1] = n[0][0]
      result[2] = n[0][1]
      bracketExpr.add(bindSym"initTable", newCall(bindSym"typeof", newEmptyNode(
         )), newCall(bindSym"typeof", newEmptyNode()))
      template adder(res, k, v) = res[k] = v
      result[0] = getAst(adder(res, n[0][0], n[0][1]))
   of nnkCurly:
      result[2] = n[0]
      bracketExpr.add(bindSym"initHashSet", newCall(bindSym"typeof",
            newEmptyNode()))
      template adder(res, v) = res.incl(v)
      result[0] = getAst(adder(res, n[0]))
   else:
      result[2] = n
      bracketExpr.add(if not hasIfs: bindSym"newSeqOfCap" else: bindSym"newSeq", newCall(
            bindSym"typeof", newEmptyNode()))
      template adder(res, v) = res.add(v)
      result[0] = getAst(adder(res, n))

macro collect*(body): untyped =
   ## Comprehension for seqs/sets/tables.
   ##
   ## The last expression of ``body`` has special syntax that specifies
   ## the collection's add operation. Use ``{e}`` for set's ``incl``,
   ## ``{k: v}`` for table's ``[]=`` and ``e`` for seq's ``add``.
   # analyse the body, find the deepest expression 'it' and replace it via
   # 'result.add it'
   let res = genSym(nskVar, "collectResult")
   let bracketExpr = newNimNode(nnkBracketExpr)
   var hasIfs = false
   var (resBody, keyType, valueType, sizeHint) = transLastStmt(body, res, bracketExpr, hasIfs)
   if bracketExpr.len == 3:
      bracketExpr[1][1] = keyType
      bracketExpr[2][1] = valueType
   else:
      bracketExpr[1][1] = valueType
   let call = newTree(nnkCall, bracketExpr)
   if not hasIfs:
      call.add sizeHint
   result = newTree(nnkStmtListExpr, newVarStmt(res, call), resBody, res)
   echo result.repr

when isMainModule:
   var data = @["bird", "word"]
   assert collect(for (i, d) in data.pairs: (if i mod 2 == 0: d)) == @["bird"]
   assert collect(for (i, d) in data.pairs: {i: d}) == {1: "word",
         0: "bird"}.toTable
   assert collect(for d in data.items: {d}) == data.toHashSet

   let y = collect:
      for (i, d) in data.pairs:
         if i mod 2 == 0: d
   assert y == @["bird"]
   assert collect((let a = 1; for (i, d) in data.pairs: (if i == a: d))) == @["word"]
   assert collect(for i in 1 ..< data.len: data[i]) == @["word"]

   let x = collect:
      for i in 0 .. data.len-1:
         for j in countdown(data.high, 0, 2):
            data[j]
   echo x
   assert x == @["bird", "word", "bird", "word"]
