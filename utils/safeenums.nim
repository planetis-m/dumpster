import std/macros, fusion/astdsl

macro toEnumImpl(x, res: typed): untyped =
  result = buildAst(stmtList):
    template raiseInvalidConv(t: untyped) =
      raise newException(ValueError, $x & " can't be converted to " & t)

    let typeSym = getTypeInst(res)
    let typeNode = getTypeImpl(typeSym)
    caseStmt(x):
      for i in 1..<typeNode.len:
        let field = typeNode[i]
        ofBranch(call(getTypeInst(x), field)):
          asgn(res, field)
      `else`:
        getAst(raiseInvalidConv($typeSym))

proc toEnum*[E: enum](x: SomeInteger, t: typedesc[E]): E {.inline.} =
  toEnumImpl(x, result)

when isMainModule:
  type
    Foo = enum
      bar, baz

  let x = toEnum(2, Foo)
