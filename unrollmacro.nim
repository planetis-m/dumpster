import macros

proc replaceIdent(node, ident: NimNode, replacement: NimNode) =
   for i in 0 ..< node.len:
      if ident == node[i]:
         node[i] = replacement

      if node[i].len >= 1:
         node[i].replaceIdent(ident, replacement)

macro unrolled(x: ForLoopStmt): untyped =
   expectKind(x, nnkForStmt)
   result = newNimNode(nnkStmtList)

   let
      loopIndex = x[0]
      loopRange = x[1][1]
      loopBody = x[^1]

   var unrollRange: Slice[int]
   case loopRange.kind
   of nnkInfix:
      let
         a = int(loopRange[1].intVal)
         b = int(loopRange[2].intVal)

      if eqIdent($loopRange[0], "..<"):
         unrollRange = a ..< b
      elif eqIdent($loopRange[0], ".."):
         unrollRange = a .. b
   else:
      error("Unsupported iterator")

   var index = newNimNode(nnkIntLit)
   loopBody.replaceIdent(loopIndex, index)
   for i in unrollRange:
      index = newLit(i)
      loopBody.copyChildrenTo(result)

   echo result.treeRepr

var someSequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
var accumulate = 0

for i in unrolled(0 ..< 10):
  echo someSequence[i]

