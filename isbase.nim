# https://forum.nim-lang.org/t/9027
import macros

macro isBaseType(t: typedesc): bool =
  let td = getTypeInst(t)
  assert td.typeKind == ntyTypeDesc
  var x = getTypeImpl(td[1])
  if x.typeKind == ntyRef:
    x = getTypeImpl(x[0])

  if x.typeKind == ntyObject:
    result = newLit(x.getTypeImpl()[1].kind == nnkEmpty)
  else:
    result = newLit(false)

type
  Base = ref object
  Obj = ref object of RootObj

echo isBaseType(Obj)
echo isBaseType(Base)
