import std/[assertions, syncio]
# Factorials

proc createFactTable[N: static[int]](): array[N, int] =
  result[0] = 1
  for i in 1 ..< N:
    result[i] = result[i - 1] * i

const factTable =
  when sizeof(int) == 4:
    createFactTable[13]()
  else:
    createFactTable[21]()

proc fact*(n: int): int =
  assert(n >= 0, $n & " must not be negative.")
  assert(n < factTable.len, $n & " is too large to look up in the table")
  factTable[n]

echo fact(6)
