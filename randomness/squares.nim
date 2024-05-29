import std/math

const key = 0xeb314a6fe49f6b17'u64

proc squares32(ctr, key: uint64): uint32 {.inline.} =
  # Widynski, Bernard (2020). "Squares: A Fast Counter-Based RNG". arXiv:2004.06278
  var x = ctr * key
  let y = x
  let z = y + key
  x = x * x + y
  x = (x shr 32) or (x shl 32) # round 1
  x = x * x + z
  x = (x shr 32) or (x shl 32) # round 2
  x = x * x + y
  x = (x shr 32) or (x shl 32) # round 3
  result = uint32((x * x + z) shr 32) # round 4

proc squares64(ctr, key: uint64): uint64 {.inline.} =
  # Widynski, Bernard (2020). "Squares: A Fast Counter-Based RNG". arXiv:2004.06278
  var x = ctr * key
  let y = x
  let z = y + key
  x = x * x + y
  x = (x shr 32) or (x shl 32) # round 1
  x = x * x + z
  x = (x shr 32) or (x shl 32) # round 2
  x = x * x + y
  x = (x shr 32) or (x shl 32) # round 3
  x = x * x + z
  let t = x
  x = (x shr 32) or (x shl 32) # round 4
  result = t xor ((x * x + y) shr 32) # round 5

proc rand*(ctr: uint64; max: range[0'f64..high(float64)]): float64 =
  let x = squares64(ctr, key)
  let u = (0x3ff'u64 shl 52'u64) or (x shr 12'u64)
  result = (cast[float64](u) - 1'f64) * max

proc rand*(ctr: uint64; max: range[0'f32..high(float32)]): float32 =
  let x = squares32(ctr, key)
  let u = (0x7f'u32 shl 23'u32) or (x shr 9'u32)
  result = (cast[float32](u) - 1'f32) * max

proc gauss*[T: SomeFloat](ctr: uint64; mu, sigma: T): T =
  # Ratio of uniforms method for normal
  # https://www2.econ.osaka-u.ac.jp/~tanizaki/class/2013/econome3/13.pdf
  const K = sqrt(2 / T(E))
  var
    a = T(0)
    b = T(0)
    ctr = ctr
  while true:
    a = rand(ctr, T(1))
    b = (T(2) * rand(ctr+1, T(1)) - T(1)) * K
    inc(ctr, 2)
    if  b * b <= -T(4) * a * a * ln(a): break
  result = mu + sigma * (b / a)

import std/stats
var rs: RunningStat
for j in 1..5:
  for i in 1 .. 100_000:
    rs.push(gauss(123456789+i.uint64*1000, 0.0f, 1.0f))
  doAssert abs(rs.mean-0) < 0.08, $rs.mean
  doAssert abs(rs.standardDeviation()-1.0) < 0.1
  let bounds = [3.5, 5.0]
  for a in [rs.max, -rs.min]:
    doAssert a >= bounds[0] and a <= bounds[1]
  rs.clear()
