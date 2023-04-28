import nimsimd/avx2

when defined(gcc) or defined(clang):
  {.localPassC: "-mavx2".}

func mul2x2Alpha(a, b: array[4, int32]): array[4, int32] =
  let
    a11 = a[0]
    a12 = a[1]
    a21 = a[2]
    a22 = a[3]
  let
    b11 = b[0]
    b12 = b[1]
    b21 = b[2]
    b22 = b[3]

  let
    hs1 = (a21 - a22) * b12
    hs2 = (a11 + a21 - a22) * (b12 + b21 + b22)
    hs3 = (a11 - a12 + a21 - a22) * (b21 + b22)
    hs4 = a12 * b21
    hs5 = (a11 + a21) * (b11 + b12 + b21 + b22)
    hs6 = a11 * b11
    hs7 = a22 * (b12 + b22)

  let c11 = hs4 + hs6
  let c12 = -hs1 + hs2 - hs3 - hs4
  let c21 = -hs2 + hs5 - hs6 - hs7
  let c22 = hs1 + hs7

  return [c11, c12, c21, c22]

func mul2x2Simd(a, b: array[4, int32]): array[4, int32] =
  let
    a11 = a[0]
    a12 = a[1]
    a21 = a[2]
    a22 = a[3]
  let
    b11 = b[0]
    b12 = b[1]
    b21 = b[2]
    b22 = b[3]

  let left = mm256_set_epi32(
    a21 - a22,
    a11 + a21 - a22,
    a11 - a12 + a21 - a22,
    a12,
    a11 + a21,
    a11,
    a22,
    0,
    )
  let right = mm256_set_epi32(
    b12,
    b12 + b21 + b22,
    b21 + b22,
    b21,
    b11 + b12 + b21 + b22,
    b11,
    b12 + b22,
    0,
    )
  let hs = mm256_mullo_epi32(left, right)

  template `[]`(a: M256i; i: int): int32 = cast[array[8, int32]](a)[i]

  let c11 = hs[4] + hs[2]
  let c12 = -hs[7] + hs[6] - hs[5] - hs[4]
  let c21 = -hs[6] + hs[3] - hs[2] - hs[1]
  let c22 = hs[7] + hs[1]

  return [c11, c12, c21, c22]

func mul2x2Naive(a, b: array[4, float32]): array[4, float32] =
  let a11 = a[0]
  let a12 = a[1]
  let a21 = a[2]
  let a22 = a[3]

  let b11 = b[0]
  let b12 = b[1]
  let b21 = b[2]
  let b22 = b[3]

  let c11 = a11 * b11 + a12 * b21
  let c12 = a11 * b12 + a12 * b22
  let c21 = a21 * b11 + a22 * b21
  let c22 = a21 * b12 + a22 * b22

  return [c11, c12, c21, c22]

proc check(a, b: array[4, int32]) =
  let c = mul2x2Simd(a, b)
  let d = mul2x2Alpha(a, b)
  assert c == d, "mul2x2 failed"

proc main() =
  # check([1'i32, 2, 3, 4], [5'i32, 6, 7, 8])
  # check([12'i32, 34, 56, 78], [90'i32, 12, 34, 56])
  # check([123'i32, 456, 789, 123], [456'i32, 789, 123, 456])
  # check([1234'i32, 5678, 9012, 3456], [7890'i32, 1234, 5678, 9012])
  # check([12345'i32, 67890, 12345, 67890], [12345'i32, 67890, 12345, 67890])
  echo mul2x2Naive([1'f32, 2, 3, 4], [5'f32, 6, 7, 8])

main()
