type
  Tag* = enum
    text, html, head, body, table, tr, th, td
  TagWithKids = range[html..high(Tag)]
  HtmlNode* = object
    case tag: Tag
    of text: s: string
    else: kids: seq[HtmlNode]

proc newTextNode*(s: sink string): HtmlNode =
  HtmlNode(tag: text, s: s)

proc newTree*(tag: TagWithKids; kids: varargs[HtmlNode]): HtmlNode =
  HtmlNode(tag: tag, kids: @kids)

proc add*(parent: var HtmlNode; kid: sink HtmlNode) =
  parent.kids.add kid

from std/xmltree import addEscaped

proc toString(n: HtmlNode; result: var string) =
  case n.tag
  of text:
    result.addEscaped n.s
  else:
    result.add "<" & $n.tag
    if n.kids.len == 0:
      result.add " />"
    else:
      result.add ">\n"
      for k in items(n.kids): toString(k, result)
      result.add "\n</" & $n.tag & ">"

proc `$`*(n: HtmlNode): string =
  result = newStringOfCap(1000)
  toString n, result

import std/macros

proc whichTag(n: NimNode): Tag =
  for e in low(TagWithKids)..high(TagWithKids):
    if n.eqIdent($e): return e
  return text

proc traverse(n, dest: NimNode): NimNode =
  if n.kind in nnkCallKinds:
    if n[0].eqIdent("text"):
      expectLen n, 2
      result = newCall(bindSym"newTextNode", n[1])
      if dest != nil:
        result = newCall(bindSym"add", dest, result)
    else:
      let tag = whichTag(n[0])
      if tag == text:
        result = copyNimNode(n)
        result.add n[0]
        for i in 1..<n.len:
          result.add traverse(n[i], nil)
      else:
        let tmpTree = genSym(nskVar, "tmpTree")
        result = newTree(nnkStmtList,
          newVarStmt(tmpTree, newCall(bindSym"newTree", n[0])))
        for i in 1..<n.len:
          result.add traverse(n[i], tmpTree)
        if dest != nil:
          result.add newCall(bindSym"add", dest, tmpTree)
  else:
    result = copyNimNode(n)
    for child in n:
      result.add traverse(child, dest)

macro buildHtml*(n: untyped): untyped =
  let tmpTree = genSym(nskVar, "tmpTree")
  let call = newCall(bindSym"newTree", bindSym"html")
  result = newTree(nnkStmtListExpr, newVarStmt(tmpTree, call))
  result.add traverse(n, tmpTree)
  result.add tmpTree

when isMainModule:
  proc toTable(headers: openarray[string]; data: seq[seq[int]]): HtmlNode =
    assert headers.len == data.len
    result = buildHtml:
      body:
        table:
          for i in 0..<data.len:
            tr:
              th:
                text headers[i]
              for col in data[i]:
                td:
                  text $col
