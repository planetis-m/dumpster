import nimsimd/avx2

when defined(gcc) or defined(clang):
  {.localPassC: "-mavx2".}

proc mul4x4Simd(a, b: array[16, int32]): array[16, int32] =
  let
    a11 = a[0]
    a12 = a[1]
    a13 = a[2]
    a14 = a[3]
    a21 = a[4]
    a22 = a[5]
    a23 = a[6]
    a24 = a[7]
    a31 = a[8]
    a32 = a[9]
    a33 = a[10]
    a34 = a[11]
    a41 = a[12]
    a42 = a[13]
    a43 = a[14]
    a44 = a[15]
  let
    b11 = b[0]
    b12 = b[1]
    b13 = b[2]
    b14 = b[3]
    b21 = b[4]
    b22 = b[5]
    b23 = b[6]
    b24 = b[7]
    b31 = b[8]
    b32 = b[9]
    b33 = b[10]
    b34 = b[11]
    b41 = b[12]
    b42 = b[13]
    b43 = b[14]
    b44 = b[15]

  let lefts = [
    mm256_set_epi32(
      (a11 + a31),
      (a11 - a13 + a31),
      -a13,
      -a33,
      -a31,
      (a11 - a13 + a31 - a33),
      (-a21 + a22 - a23 - a24),
      (-a21 + a22 - a23 - a24 - a41 + a42),
    ),
    mm256_set_epi32(
      (a11 - a13),
      (-a21 + a22 - a41 + a42),
      (a41 - a42),
      (-a21 + a22 - a23 - a24 - a41 + a42 - a43 - a44),
      (-a23 - a24),
      (a11 - a12 + a21 - a22),
      (-a12 - a14),
      (a12 + a14 - a21 + a22 + a23 + a24),
    ),
    mm256_set_epi32(
      (a12 + a14 - a21 + a22 + a23 + a24 + a32 + a41 - a42),
      (a12 - a21 + a22 + a32 + a41 - a42),
      (a14 + a23 + a24),
      (a12 + a14 - a21 + a22 + a23 + a24 + a32 + a34 + a41 - a42 - a43 - a44),
      (a32 + a41 - a42),
      (a12 + a14 + a22 + a24),
      (a12 + a14 + a22 + a24 + a32 - a42),
      (a14 + a24),
    ),
    mm256_set_epi32(
      (a12 + a14 + a22 + a24 + a32 + a34 - a42 - a44),
      (a32 - a42),
      (a34 - a44),
      (a34 - a43 - a44),
      (a14 + a34),
      (a13 + a14 + a23 + a24 + a33 + a34 - a43 - a44),
      (a11 - a12 - a13 - a14 + a21 - a22 - a23 - a24 + a31 - a32 - a33 - a34 - a41 + a42 + a43 +
       a44),
      -a43,
    ),
    mm256_set_epi32(
      a14,
      (a14 - a32),
      (a13 + a14 + a23 + a24 - a31 + a32 + a33 + a34 + a41 - a42 - a43 - a44),
      (-a31 + a32 + a33 + a34 + a41 - a42 - a43 - a44),
      (-a12 - a32),
      (a32 + a34),
      (-a13 - a14 - a23 - a24),
      a32,
    ),
    mm256_set_epi32(
      -a21,
      (-a21 + a41),
      (-a21 + a41 - a43),
      (a12 + a22 + a32 - a42),
      (-a21 + a23 + a41 - a43),
      (-a31 + a32 + a41 - a42),
      (a41 - a43),
      (-a43 - a44),
    ),
    mm256_set_epi32(-a23, 0, 0, 0, 0, 0, 0, 0),
  ]
  let rights = [
    mm256_set_epi32(
      (b11 + b31),
      (b11 - b13 + b31),
      (b11 - b13 + b31 - b33),
      -b33,
      -b13,
      -b31,
      (-b21 + b22 - b23 - b24),
      (-b21 + b22 - b23 - b24 - b41 + b42),
    ),
    mm256_set_epi32(
      (b11 - b13),
      (-b21 + b22 - b41 + b42),
      (-b23 - b24),
      (b41 - b42),
      (-b21 + b22 - b23 - b24 - b41 + b42 - b43 - b44),
      (-b12 - b14),
      -b21,
      (b12 + b14 - b21 + b22 + b23 + b24),
    ),
    mm256_set_epi32(
      (b12 + b14 - b21 + b22 + b23 + b24 + b32 + b41 - b42),
      (b12 - b21 + b22 + b32 + b41 - b42),
      (b12 + b14 - b21 + b22 + b23 + b24 + b32 + b34 + b41 - b42 - b43 - b44),
      (b32 + b41 - b42),
      (b14 + b23 + b24),
      (b12 + b14 + b22 + b24),
      (b12 + b14 + b22 + b24 + b32 - b42),
      (b12 + b14 + b22 + b24 + b32 + b34 - b42 - b44),
    ),
    mm256_set_epi32(
      (b32 - b42),
      (b14 + b24),
      (b34 - b44),
      (b34 - b43 - b44),
      -b43,
      (b14 + b34),
      b14,
      (b13 + b14 + b23 + b24 + b33 + b34 - b43 - b44),
    ),
    mm256_set_epi32(
      (-b21 + b41),
      (-b21 + b41 - b43),
      (b14 - b32),
      b32,
      -b23,
      (b41 - b43),
      (b32 + b34),
      (-b21 + b23 + b41 - b43),
    ),
    mm256_set_epi32(
      (b11 - b12 + b21 - b22),
      (b11 - b12 - b13 - b14 + b21 - b22 - b23 - b24 + b31 - b32 - b33 - b34 - b41 + b42 + b43 +
       b44),
      (b13 + b14 + b23 + b24 - b31 + b32 + b33 + b34 + b41 - b42 - b43 - b44),
      (b12 + b22 + b32 - b42),
      (-b31 + b32 + b33 + b34 + b41 - b42 - b43 - b44),
      (-b12 - b32),
      (-b13 - b14 - b23 - b24),
      (-b43 - b44),
    ),
    mm256_set_epi32((-b31 + b32 + b41 - b42), 0, 0, 0, 0, 0, 0, 0),
  ]
  let hs = [
    mm256_mullo_epi32(lefts[0], rights[0]),
    mm256_mullo_epi32(lefts[1], rights[1]),
    mm256_mullo_epi32(lefts[2], rights[2]),
    mm256_mullo_epi32(lefts[3], rights[3]),
    mm256_mullo_epi32(lefts[4], rights[4]),
    mm256_mullo_epi32(lefts[5], rights[5]),
    mm256_mullo_epi32(lefts[6], rights[6]),
  ]

  template `[]`(a: M256i; i: int): int32 = cast[array[8, int32]](a)[i]

  let c11 = (hs[0][7] - hs[0][6] - hs[0][3] + hs[1][7] + hs[1][1] + hs[4][7])
  let c12 = (-hs[0][1] + hs[0][0] - hs[1][6] + hs[1][5] - hs[1][2] + hs[1][1] + hs[1][0] -
              hs[2][7] + hs[2][6] + hs[2][3] - hs[3][1] + hs[4][7] - hs[4][5] - hs[4][4])
  let c13 =
    (hs[0][7] - hs[0][6] + hs[0][5] - hs[0][3] + hs[4][7] - hs[4][6] + hs[4][3] - hs[4][0])
  let c14 =
    (hs[0][0] - hs[1][6] + hs[1][5] - hs[1][3] + hs[2][7] - hs[2][6] - hs[2][5] - hs[2][3] +
     hs[3][1] - hs[4][7] + hs[4][6] + hs[4][5] + hs[4][4] - hs[4][3] - hs[4][1] + hs[4][0])
  let c21 = (-hs[1][1] - hs[1][0] + hs[2][7] - hs[2][6] - hs[2][3] + hs[2][2] -
              hs[2][1] + hs[3][6] - hs[4][7] - hs[5][7] + hs[5][4] + hs[6][7])
  let c22 = (hs[0][1] - hs[0][0] + hs[1][6] - hs[1][5] - hs[1][1] - hs[1][0] + hs[2][7] -
             hs[2][6] - hs[2][3] + hs[2][2] - hs[2][1] + hs[3][6] - hs[4][7] + hs[5][4])
  let c23 = (hs[2][7] - hs[2][6] - hs[2][5] - hs[2][3] - hs[2][1] + hs[2][0] + hs[3][6] - hs[4][7] +
             hs[4][6] - hs[4][3] + hs[4][0] - hs[5][5] + hs[5][4] + hs[5][3] - hs[5][1] + hs[6][7])
  let c24 = (-hs[0][0] + hs[1][6] - hs[1][5] + hs[1][3] - hs[2][7] + hs[2][6] + hs[2][5] + hs[2][3] +
              hs[2][1] - hs[2][0] - hs[3][6] + hs[4][7] - hs[4][6] + hs[4][3] - hs[4][0] - hs[5][4])
  let c31 =
    (hs[0][6] + hs[0][3] + hs[0][2] - hs[1][7] - hs[3][3] - hs[4][7] + hs[4][6] + hs[4][2])
  let c32 = (-hs[0][1] + hs[0][0] + hs[1][5] + hs[1][4] - hs[1][0] + hs[2][7] - hs[2][4] -
              hs[2][3] - hs[3][3] - hs[4][7] + hs[4][6] + hs[4][4] + hs[4][2] + hs[5][2])
  let c33 = (hs[0][4] + hs[0][3] - hs[3][3] - hs[4][7] + hs[4][6] + hs[4][0])
  let c34 = (hs[1][5] + hs[2][3] - hs[3][4] + hs[3][3] + hs[3][2] + hs[4][7] -
             hs[4][6] - hs[4][5] - hs[4][4] + hs[4][1] - hs[4][0] + hs[5][0])
  let c41 =
    (-hs[1][0] + hs[2][7] - hs[2][4] - hs[2][3] + hs[2][2] - hs[2][1] + hs[3][7] + hs[3][6] -
      hs[3][3] - hs[3][0] - hs[4][7] + hs[4][6] + hs[4][2] - hs[5][7] + hs[5][6] + hs[5][5])
  let c42 =
    (-hs[0][1] + hs[0][0] + hs[1][5] + hs[1][4] - hs[1][0] + hs[2][7] - hs[2][4] - hs[2][3] +
      hs[2][2] - hs[2][1] + hs[3][7] + hs[3][6] - hs[3][3] - hs[4][7] + hs[4][6] + hs[4][2])
  let c43 = (-hs[2][3] + hs[3][6] - hs[3][5] + hs[3][4] - hs[3][3] -
              hs[3][0] - hs[4][7] + hs[4][6] + hs[4][0] - hs[5][1])
  let c44 = (hs[1][5] + hs[2][3] - hs[3][6] + hs[3][5] - hs[3][4] +
             hs[3][3] + hs[4][7] - hs[4][6] - hs[4][0] + hs[5][0])

  return [
    c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34, c41, c42, c43, c44,
  ]

proc mul4x4Naive(a, b: array[16, int32]): array[16, int32] =
  let
    a11 = a[0]
    a12 = a[1]
    a13 = a[2]
    a14 = a[3]
    a21 = a[4]
    a22 = a[5]
    a23 = a[6]
    a24 = a[7]
    a31 = a[8]
    a32 = a[9]
    a33 = a[10]
    a34 = a[11]
    a41 = a[12]
    a42 = a[13]
    a43 = a[14]
    a44 = a[15]
  let
    b11 = b[0]
    b12 = b[1]
    b13 = b[2]
    b14 = b[3]
    b21 = b[4]
    b22 = b[5]
    b23 = b[6]
    b24 = b[7]
    b31 = b[8]
    b32 = b[9]
    b33 = b[10]
    b34 = b[11]
    b41 = b[12]
    b42 = b[13]
    b43 = b[14]
    b44 = b[15]

  let c11 = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41
  let c12 = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42
  let c13 = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43
  let c14 = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44
  let c21 = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41
  let c22 = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42
  let c23 = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43
  let c24 = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44
  let c31 = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41
  let c32 = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42
  let c33 = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43
  let c34 = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44
  let c41 = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41
  let c42 = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42
  let c43 = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43
  let c44 = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44

  return [
    c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34, c41, c42, c43, c44,
  ]

proc mul4x4Iter2(a, b: array[16, int32]): array[16, int32] =
  for i in 0..3:
    for j in 0..3:
      var tmp: int32 = 0
      for k in 0..3:
        tmp += a[i * 4 + k] * b[k * 4 + j]
      result[i * 4 + j] = tmp

proc mul4x4Iter(a, b: array[16, int32]): array[16, int32] =
  var c: array[16, int32]
  for i in 0..3:
    for j in 0..3:
      var tmp: int32 = 0
      for k in 0..3:
        tmp += a[i * 4 + k] * b[k * 4 + j]
      c[i * 4 + j] = tmp
  return c

proc mul4x4Temp(a, b: array[16, int32]): array[16, int32] =
  var c: array[16, int32]
  for j in 0..3:
    var bColj: array[4, int32]
    for k in 0..3:
      bColj[k] = b[k * 4 + j]
    for i in 0..3:
      var s: int32 = 0
      for k in 0..3:
        s += a[i * 4 + k] * bColj[k]
      c[i * 4 + j] = s
  return c

# Some test cases for the naive algorithm
# echo mul4x4Simd([1'i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
#                 [1'i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

import std/[times, stats, strformat, random]

proc warmup() =
  # Warmup - make sure cpu is on max perf
  let start = cpuTime()
  var a = 123
  for i in 0 ..< 300_000_000:
    a += i * i mod 456
    a = a mod 789
  let dur = cpuTime() - start
  echo &"Warmup: {dur:>4.4f} s ", a

proc printStats(name: string, stats: RunningStat, dur: float) =
  echo &"""{name}:
  Collected {stats.n} samples in {dur:>4.4f} s
  Average time: {stats.mean * 1000:>4.4f} ms
  Stddev  time: {stats.standardDeviationS * 1000:>4.4f} ms
  Min     time: {stats.min * 1000:>4.4f} ms
  Max     time: {stats.max * 1000:>4.4f} ms"""

template bench(name, samples, code: untyped) =
  var stats: RunningStat
  let globalStart = cpuTime()
  for i in 0 ..< samples:
    let start = cpuTime()
    code
    let duration = cpuTime() - start
    stats.push duration
  let globalDuration = cpuTime() - globalStart
  printStats(name, stats, globalDuration)

const maxIters = 100_000

proc main =
  warmup()
  var a, b = [1'i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
  var d: array[16, int32]
  shuffle(a)
  shuffle(b)
  bench("mul4x4Iter", maxIters):
    d = mul4x4Iter(a, b)
  echo d
  shuffle(a)
  shuffle(b)
  bench("mul4x4Iter2", maxIters):
    d = mul4x4Iter2(a, b)
  echo d
  shuffle(a)
  shuffle(b)
  bench("mul4x4Naive", maxIters):
    d = mul4x4Naive(a, b)
  echo d
  shuffle(a)
  shuffle(b)
  bench("mul4x4Simd", maxIters):
    d = mul4x4Simd(a, b)
  echo d
  shuffle(a)
  shuffle(b)
  bench("mul4x4Temp", maxIters):
    d = mul4x4Temp(a, b)
  echo d

main()
