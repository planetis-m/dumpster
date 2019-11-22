import macros, tables, sets

proc createAst(n, res: NimNode): (NimNode, NimNode) =
   if n.kind == nnkCurly:
      expectLen(n, 1)
      let elem = n[0]
      template constr(elemType) = initHashSet[typeof(elemType)]()
      template adder(hashset, elem) = hashset.incl(elem)
      let elemType = elem
      result = (getAst(adder(res, elem)), getAst(constr(elemType)))
   elif n.kind == nnkTableConstr and n[0].kind == nnkExprColonExpr:
      expectLen(n[0], 2)
      let key = n[0][0]
      let val = n[0][1]
      template adder(table, key, val) = table[key] = val
      let keyType = key
      let valueType = val
      result = (getAst(adder(res, key, val)), getAst(constr(keyType, valueType)))
   else: # an ident
      let elem = n
      template adder(sequence, elem) = sequence.add(elem)
      let elemType = elem
      result = (getAst(adder(res, elem)), getAst(constr(elemType)))
#    echo result[1].repr

proc transLastStmt(n, res: NimNode): (NimNode, NimNode) =
   # Looks for the last statement of the last statement, etc...
   case n.kind
   of nnkStmtList, nnkStmtListExpr, nnkBlockStmt, nnkBlockExpr,
         nnkWhileStmt,
         nnkForStmt, nnkIfExpr, nnkIfStmt, nnkTryStmt, nnkCaseStmt,
         nnkElifBranch, nnkElse, nnkElifExpr:
      result[0] = copyNimTree(n)
      result[1] = copyNimTree(n)
      if n.len >= 1:
         (result[0][^1], result[1][^1]) = transLastStmt(n[^1], res)
   else:
      result = createAst(n, res)

macro comp*(body: untyped): untyped =
   # analyse the body, find the deepest expression 'it' and replace it via
   # 'result.add it'
   let res = genSym(nskVar, "collectResult")
   let (call, typeRes) = transLastStmt(body, res)
   echo typeRes.repr

   template constr(keyType, valueType) = initTable[typeof(keyType), typeof(valueType)]()

   template constr(elemType) = newSeq[typeof(elemType)]()
   getAst(constr(elemType))
   result = nnkStmtListExpr.newTree(newVarStmt(res, typeRes), call, res)
   echo repr(result)

when isMainModule:
   var data = @["bird", "word"]
   echo comp(for i, d in data.pairs: (if i mod 2 == 0: d))
   echo comp(for i, d in data.pairs: {i: d})
