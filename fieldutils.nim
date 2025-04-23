import std/macros

macro `<-`*(vars, obj: untyped): untyped =
  expectKind(vars, nnkTupleConstr)
  result = newStmtList()
  for v in vars:
    result.add newLetStmt(v, newDotExpr(obj, v))

macro `^`*(T: typedesc, vars: untyped): untyped =
  expectKind(vars, nnkTupleConstr)
  result = newTree(nnkObjConstr, T)
  for v in vars:
    result.add newColonExpr(v, v)

when isMainModule:
  type
    Person = object
      name: string
      age: int

  var p = Person(name: "Alice", age: 30)
  (let name, age) <- p

  p = Person^(name, age)
