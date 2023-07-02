import std/bitops

type
  ElemType = uint

const
  ElemSize = sizeof(ElemType)*8

  One = ElemType(1)
  Zero = ElemType(0)

proc enumRange(t: typedesc): int =
  high(t).ord - low(t).ord + 1

proc wordsFor(t: typedesc): int =
  result = (enumRange(t) + ElemSize - 1) div ElemSize

type
  BitSet*[T: enum] = object
    a: array[wordsFor(ElemType), ElemType]

template modElemSize(arg: untyped): untyped = arg.ord and (ElemSize - 1)
template divElemSize(arg: untyped): untyped = arg.ord shr countTrailingZeroBits(ElemSize)

template `[]`(x: BitSet, i: int): ElemType = x.a[i]
template `[]=`(x: BitSet, i: int, v: ElemType) =
  x.a[i] = v

proc contains*[T](x: BitSet[T], e: T): bool =
  result = (x[int(e.divElemSize)] and (One shl e.modElemSize)) != Zero

proc incl*[T](x: var BitSet[T], elem: T) =
  x[int(elem.divElemSize)] = x[int(elem.divElemSize)] or
      (One shl elem.modElemSize)

proc excl*[T](x: var BitSet[T], elem: T) =
  x[int(elem.divElemSize)] = x[int(elem.divElemSize)] and
      not(One shl elem.modElemSize)

proc union*[T](x: var BitSet[T], y: BitSet[T]) =
  for i in 0..high(x): x[i] = x[i] or y[i]

proc diff*[T](x: var BitSet[T], y: BitSet[T]) =
  for i in 0..high(x): x[i] = x[i] and not y[i]

proc symDiff*[T](x: var BitSet[T], y: BitSet[T]) =
  for i in 0..high(x): x[i] = x[i] xor y[i]

proc intersect*[T](x: var BitSet[T], y: BitSet[T]) =
  for i in 0..high(x): x[i] = x[i] and y[i]

proc equals*[T](x, y: BitSet[T]): bool =
  result = true
  for i in 0..high(x):
    if x[i] != y[i]:
      return false

proc contains*[T](x, y: BitSet[T]): bool =
  result = true
  for i in 0..high(x):
    if (y[i] and not x[i]) != Zero:
      return false

proc `*`*[T](x, y: BitSet[T]): BitSet[T] {.inline.} = (var x = x; intersect(x, y))
proc `+`*[T](x, y: BitSet[T]): BitSet[T] {.inline.} = (var x = x; union(x, y))
proc `-`*[T](x, y: BitSet[T]): BitSet[T] {.inline.} = (var x = x; diff(x, y))
proc `<`*[T](x, y: BitSet[T]): bool {.inline.} = contains(y, x) and not equals(x, y)
proc `<=`*[T](x, y: BitSet[T]): bool {.inline.} = contains(y, x)
proc `==`*[T](x, y: BitSet[T]): bool {.inline.} = equals(x, y)

when isMainModule:
  type
    HasComponent = enum
      HasCollide, HasDirty, HasDraw2d, HasFade, HasHierarchy,
      HasMove, HasShake, HasTransform2d
    Signature = BitSet[HasComponent]

  proc main =
    var sig: Signature
    echo sig.a.len # 0
    sig.incl HasDirty
    echo HasDirty in sig
    echo HasShake in sig

  main()
