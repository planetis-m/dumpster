# jackhftang https://forum.nim-lang.org/t/11893#75562
from std/bitops import countLeadingZeroBits

proc log2(x: int): int {.inline.} =
  # Undefined for zero argument
  result = sizeof(int)*8 - 1 - countLeadingZeroBits(x)

proc pow2(power: int): int {.inline.} =
  # Raise 2 into the specified power.
  result = 1 shl power

type
  StableSeq*[T] = object
    len: int
    # sizeof(data[i]) == pow2(i)
    data: seq[ptr UncheckedArray[T]]

proc add*[T](s: var StableSeq[T], v: sink T): ptr T =
  let i = log2(s.len+1)
  let j = s.len+1 - pow2(i)
  while i >= s.data.len:
    let p = cast[ptr UncheckedArray[T]](alloc0(sizeof(T)*pow2(s.data.len)))
    s.data.add p
  inc s.len
  result = s.data[i][j].addr
  result[] = ensureMove(v)

proc `[]`*[T](s: var StableSeq[T], k: int): var T =
  let i = log2(k+1)
  let j = k+1 - pow2(i)
  return s.data[i][j]

proc `[]=`*[T](s: var StableSeq[T], k: int, v: sink T) =
  let i = log2(k+1)
  let j = k+1 - pow2(i)
  s.data[i][j] = v

when isMainModule:
  var s: StableSeq[int]
  for i in 0 .. 10:
    discard s.add(i)
  s[5] = 15
  for i in 0 ..< s.len:
    echo i, ' ', s[i]
