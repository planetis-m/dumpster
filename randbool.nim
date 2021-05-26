import random

const
  Scale = 2.0 * float(1'u64 shl 63) # wrong!!

proc genBool(r: var Rand; p: range[0.0..1.0]): bool =
  result = if p == 1: true else: r.next() < uint64(p * Scale)

proc genBool(p: float): bool {.inline.} =
  result = genBool(randState, p)

echo genBool(1 / 4)
