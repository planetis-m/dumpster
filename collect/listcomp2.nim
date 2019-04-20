import macros, sequtils, future

type 
  Container[T] = concept x
    for item in x:
      item is T

proc forCompImpl(yieldResult: bool, comp: NimNode): NimNode =
  expectLen(comp, 3)
  expectKind(comp, nnkInfix)
  expectKind(comp[0], nnkIdent)
  assert($comp[0].ident == "|")

  result = comp[1]
  var yieldNow = yieldResult

  for i in countdown(comp[2].len-1, 0):
    var x = comp[2][i]
    if x.kind != nnkInfix or $x[0] != "<-":
      x = newNimNode(nnkInfix).add(ident"<-").add(ident"_").add(x)
    expectLen(x, 3)
    var iDef: NimNode
    var iType: NimNode
    let elem = x[2]
    let elemType = quote do:
      # If we have elemType proc (or template) - use it
      when compiles(elemType(`elem`)):
        elemType(`elem`)
      # Otherwise we get type ourselves
      elif `elem` is Container:
        type(`elem`[0])
      else:
        type(`elem`)
    
    if x[1].kind == nnkIdent:
      iDef = x[1]
      iType = elemType
    else:
      expectLen(x[1], 1)
      expectMinLen(x[1][0], 2)
      expectKind(x[1][0][0], nnkIdent)
      iDef = x[1][0][0]
      iType = x[1][0][1]
    let cont = x[2]
    let lmb = newProc(
      params = @[ident"auto", newIdentDefs(iDef, iType)], 
      body = result, 
      procType = nnkLambda
    )
    let p = newNimNode(nnkPragma)
    p.add(ident"closure")
    lmb[4] = p
    if yieldNow:
      yieldNow = false
      result = quote do:
        `cont`.map(`lmb`)
    else:
      result = quote do:
        `cont`.flatmap(`lmb`)

macro lc*(comp: untyped): untyped =
  ## For comprehension with Haskell ``do notation`` like syntax.
  ## Example:
  ##
  ## .. code-block:: nim
  ##
  ##   let res = act do:
  ##     (x: int) <- 1.some,
  ##     (y: int) <- (x + 3).some
  ##     (y*100).some
  ##   assert(res == 400.some)
  ##
  ## The only requirement for the user is to implement `foldMap`` function for the type
  ##
  expectKind comp, {nnkStmtList, nnkDo}
  let stmts = if comp.kind == nnkStmtList: comp else: comp.findChild(it.kind == nnkStmtList)
  expectMinLen(stmts, 2)
  let op = newNimNode(nnkInfix)
  op.add(ident"|")
  let res = stmts[stmts.len-1]
  var yieldResult = false
  if res.kind == nnkYieldStmt:
    yieldResult = true
    op.add(res[0].copyNimTree)
  else:
    op.add(res.copyNimTree)
  let par = newNimNode(nnkPar)
  op.add(par)
  for i in 0..<(stmts.len-1):
    par.add(stmts[i].copyNimTree)

  result = forCompImpl(yieldResult, op)
  echo result.repr

proc flatMap[T](s: seq[T], f: T -> seq[T]): seq[T] {.inline.} =
  result = newSeqOfCap[T](s.len)
  for v in s:
    result.add(f(v))

let res = lc:
  x <- @[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  y <- @[100, 200, 300]
  z <- @[5, 7]
  @[x * y + z]

echo res
