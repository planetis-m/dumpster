import std/[macros, sets, strutils]

const
  SpecialAttrs = ["intval", "floatval", "strval"]

var
  allnimnodes {.compileTime.}: HashSet[string]

proc isNimNode(x: string): bool {.compileTime.} =
  allnimnodes.contains(x)

proc addNodes() {.compileTime.} =
  for i in nnkEmpty..NimNodeKind.high:
    allnimnodes.incl normalize(substr($i, 3))
  allnimnodes.excl "ident"

static:
  addNodes()

proc getName(n: NimNode): string =
  case n.kind
  of nnkStrLit..nnkTripleStrLit, nnkIdent, nnkSym:
    result = n.strVal
  of nnkDotExpr:
    result = getName(n[1])
  of nnkAccQuoted, nnkOpenSymChoice, nnkClosedSymChoice:
    result = getName(n[0])
  else:
    expectKind(n, nnkIdent)

proc newDotAsgn(tmp: NimNode, key: string, x: NimNode): NimNode =
  result = newTree(nnkAsgn, newDotExpr(tmp, newIdentNode key), x)

proc traverse(n, dest: NimNode): NimNode =
  if n.kind in nnkCallKinds - {nnkInfix}:
    let op = normalize(getName(n[0]))
    if isNimNode(op):
      let tmpTree = genSym(nskVar, "tmpTree")
      result = newTree(nnkStmtList,
        newVarStmt(tmpTree, newCall(bindSym"newNimNode", ident("nnk" & op))))
      for i in 1..<n.len:
        let x = n[i]
        if x.kind == nnkExprEqExpr:
          let key = normalize(getName(x[0]))
          if key in SpecialAttrs:
            result.add newDotAsgn(tmpTree, key, x[1])
          else: error("Unsupported setter: " & key, x)
        else:
          result.add traverse(x, tmpTree)
      if dest != nil:
        result.add newCall(bindSym"add", dest, tmpTree)
    elif op == "ident":
      expectLen n, 2
      if dest != nil:
        result = newCall(bindSym"add", dest, n)
    elif op == "!" and n.len == 2:
      if dest != nil:
        result = newCall(bindSym"add", dest, n[1])
    else:
      result = copyNimNode(n)
      result.add n[0]
      for i in 1..<n.len:
        result.add traverse(n[i], nil)
  else:
    result = copyNimNode(n)
    for child in n:
      result.add traverse(child, dest)

macro buildAst*(n: untyped): untyped =
  let tmpTree = genSym(nskVar, "tmpTree")
  let call = newCall(bindSym"newNimNode", bindSym"nnkStmtList")
  result = newTree(nnkStmtListExpr, newVarStmt(tmpTree, call))
  result.add traverse(n, tmpTree)
  result.add tmpTree
  echo result.repr

when isMainModule:
  template templ1(e) {.dirty.} =
    var e = 2
    echo(e + 2)
  macro test1: untyped =
    let e = genSym(nskVar, "e")
    result = buildAst:
      !newVarStmt(e, newLit(2))
      call(ident"echo"):
        infix(ident("+"), !e, intLit(intVal = 2))
    assert result == getAst(templ1(e))
  test1()
