import macros

proc createAst(n: NimNode): (NimNode, NimNode) =
   if n.kind == nnkCurly:
      expectLen(n, 1)
      let elem = n[0]
      template constr(elemType) = initHashSet[typeof(elemType)]()
      template adder(hashset, elem) = hashset.incl(elem)
      let elemType = getStmt(elem)
      result = (getStmt(getAst(adder(retVar, elem))), getAst(constr(elemType)))
   elif n.kind == nnkTableConstr and n[0].kind == nnkExprColonExpr:
      expectLen(n[0], 2)
      let key = n[0][0]
      let val = n[0][1]
      template constr(keyType, valueType) = initTable[typeof(keyType), typeof(valueType)]()
      template adder(table, key, val) = table[key] = val
      let keyType = getStmt(key)
      let valueType = getStmt(val)
      result = (getStmt(getAst(adder(retVar, key, val))),
         getAst(constr(keyType, valueType)))
   else: # an ident
      let elem = n
      template constr(elemType) = newSeq[typeof(elemType)]()
      template adder(sequence, elem) = sequence.add(elem)
      let elemType = getStmt(elem)
      result = (getStmt(getAst(adder(retVar, elem))), getAst(constr(elemType)))

proc transLastStmt(n: NimNode): NimNode =
   # Looks for the last statement of the last statement, etc...
   case n.kind
   of nnkStmtList, nnkStmtListExpr, nnkBlockStmt, nnkBlockExpr,
         nnkWhileStmt,
         nnkForStmt, nnkIfExpr, nnkIfStmt, nnkTryStmt, nnkCaseStmt,
         nnkElifBranch, nnkElse, nnkElifExpr:
      result = copyNimTree(n)
      if n.len >= 1:
         result[^1] = transLastStmt(n[^1], b)
   else:
      result = createAst(n)

macro comp*(body: untyped): untyped =
   # analyse the body, find the deepest expression 'it' and replace it via
   # 'result.add it'
   let retVar = genSym(nskVar, "collectResult")
   let (retBody, callConstr) = transLastStmt(body)
   let tempAsgn = newTree(nnkVarSection,
      newIdentDefs(retVar, newEmptyNode(), callConstr))
   result = nnkStmtListExpr.newTree(tempAssn, retBody, retVar)
   echo repr(result)

when isMainModule:
   import tables, sets

   var data = {2: "bird", 5: "word"}.toTable
   echo comp(for k, v in data: (if k mod 2 == 0: v))
   echo comp(for k, v in data: (if k mod 2 == 0: {k: v}))
