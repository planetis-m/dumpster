# Example 1:

var
  a: array[256, cint]
  b: array[256, cint]
  c: array[256, cint]

proc foo() =
  var i: cint

  i = 0
  while i < 256:
    a[i] = b[i] + c[i]
    inc(i)

# Example 2:

var
  a: array[256, cint]
  b: array[256, cint]
  c: array[256, cint]

proc foo(n: cint; x: cint) =
  var i: cint

  # feature: support for unknown loop bound
  # feature: support for loop invariants
  i = 0
  while i < n:
    b[i] = x
    inc(i)


  # feature: general loop exit condition
  # feature: support for bitwise operations
  while dec(n):
    a[i] = b[i] and c[i]
    inc(i)

# Example 3:

type
  aint = cint

proc foo(n: cint; p: ptr aint; q: ptr aint) =
  # feature: support for (aligned) pointer accesses.
  while dec(n):
    inc(p)[] = inc(q)[]

# Example 4:

type
  aint = cint

var
  a: array[256, cint]
  b: array[256, cint]
  c: array[256, cint]

proc foo(n: cint; p: ptr aint; q: ptr aint) =
  var i: cint

  # feature: support for (aligned) pointer accesses
  # feature: support for constants
  while dec(n):
    inc(p)[] = inc(q)[] + 5


  # feature: support for read accesses with a compile time known misalignment
  i = 0
  while i < n:
    a[i] = b[i + 1] + c[i + 3]
    inc(i)


  # feature: support for if-conversion
  i = 0
  while i < n:
    j = a[i]
    b[i] = (if j > MAX: MAX else: 0)
    inc(i)

# Example 5:

type
  a {.bycopy.} = object
    ca: array[N, cint]


var s: a

i = 0
while i < N:
  # feature: support for alignable struct access
  s.ca[i] = 5
  inc(i)
# Example 6: gfortran:
# DIMENSION A(1000000), B(1000000), C(1000000)
# READ, X, Y
# A = LOG(X); B = LOG(Y); C = A + B
# PRINT, C(500000)
# END
# Example 7:

var
  a: array[256, cint]
  b: array[256, cint]

proc foo(x: cint) =
  var i: cint

  # feature: support for read accesses with an unknown misalignment
  i = 0
  while i < N:
    a[i] = b[i + x]
    inc(i)

# Example 8:

var a: array[M, array[N, cint]]

proc foo(x: cint) =
  var
    i: cint
    j: cint

  # feature: support for multidimensional arrays
  i = 0
  while i < M:
    j = 0
    while j < N:
      a[i][j] = x
      inc(j)
    inc(i)

# Example 9:

var
  ub: array[N, cuint]
  uc: array[N, cuint]

proc foo() =
  var i: cint


  # feature: support summation reduction.
  #     note: in case of floats use -funsafe-math-optimizations
  var diff: cuint = 0
  i = 0
  while i < N:
    inc(udiff, (ub[i] - uc[i]))
    inc(i)


  # Example 10:
  # feature: support data-types of different sizes.
  #   Currently only a single vector-size per target is supported; 
  #   it can accommodate n elements such that n = vector-size/element-size 
  #   (e.g, 4 ints, 8 shorts, or 16 chars for a vector of size 16 bytes). 
  #   A combination of data-types of different sizes in the same loop 
  #   requires special handling. This support is now present in mainline,
  #   and also includes support for type conversions.
  var
    sa: ptr cshort
    sb: ptr cshort
    sc: ptr cshort
  var
    ia: ptr cint
    ib: ptr cint
    ic: ptr cint
  i = 0
  while i < N:
    ia[i] = ib[i] + ic[i]
    sa[i] = sb[i] + sc[i]
    inc(i)


  i = 0
  while i < N:
    ia[i] = cast[cint](sb[i])
    inc(i)


  # Example 11:
  # feature: support strided accesses - the data elements
  #   that are to be operated upon in parallel are not consecutive - they
  #   are accessed with a stride > 1 (in the example, the stride is 2):
  i = 0
  while i < N div 2:
    a[i] = b[2  i + 1]  c[2  i + 1] - b[2  i]  c[2  i]
    d[i] = b[2  i]  c[2  i + 1] + b[2  i + 1]  c[2  i]
    inc(i)


  # Example 12: Induction:
  i = 0
  while i < N:
    a[i] = i
    inc(i)


  # Example 13: Outer-loop:
  i = 0
  while i < M:
    diff = 0
    j = 0
    while j < N:
      inc(diff, (a[i][j] - b[i][j]))
      inc(j, 8)

    `out`[i] = diff
    inc(i)

# Example 14: Double reduction:

k = 0
while k < K:
  sum = 0
  j = 0
  while j < M:
    i = 0
    while i < N:
      inc(sum, `in`[i + k][j]  coeff[i][j])
      inc(i)
    inc(j)

  `out`[k] = sum
  inc(k)
# Example 15: Condition in nested loop:

j = 0
while j < M:
  x = x_in[j]
  curr_a = a[0]

  i = 0
  while i < N:
    next_a = a[i + 1]
    curr_a = if x > c[i]: curr_a else: next_a
    inc(i)


  x_out[j] = curr_a
  inc(j)
# Example 16: Load permutation in loop-aware SLP:

i = 0
while i < N:
  a = inc(pInput)[]
  b = inc(pInput)[]
  c = inc(pInput)[]

  inc(pOutput)[] = M00  a + M01  b + M02  c
  inc(pOutput)[] = M10  a + M11  b + M12  c
  inc(pOutput)[] = M20  a + M21  b + M22  c
  inc(i)
# Example 17: Basic block SLP:

proc foo() =
  var pin: ptr cuint = addr(`in`[0])
  var pout: ptr cuint = addr(`out`[0])

  inc(pout)[] = inc(pin)[]
  inc(pout)[] = inc(pin)[]
  inc(pout)[] = inc(pin)[]
  inc(pout)[] = inc(pin)[]

# Example 18: Simple reduction in SLP:

var sum1: cint

var sum2: cint

var a: array[128, cint]

proc foo() =
  var i: cint

  i = 0
  while i < 64:
    inc(sum1, a[2  i])
    inc(sum2, a[2  i + 1])
    inc(i)

# Example 19: Reduction chain in SLP:

var sum: cint

var a: array[128, cint]

proc foo() =
  var i: cint

  i = 0
  while i < 64:
    inc(sum, a[2  i])
    inc(sum, a[2  i + 1])
    inc(i)

# Example 20: Basic block SLP with multiple types, loads with different offsets, misaligned load, and not-affine accesses:

proc foo(dst: ptr cint; src: ptr cshort; h: cint; stride: cint; A: cshort; B: cshort) =
  var i: cint
  i = 0
  while i < h:
    inc(dst[0], A  src[0] + B  src[1])
    inc(dst[1], A  src[1] + B  src[2])
    inc(dst[2], A  src[2] + B  src[3])
    inc(dst[3], A  src[3] + B  src[4])
    inc(dst[4], A  src[4] + B  src[5])
    inc(dst[5], A  src[5] + B  src[6])
    inc(dst[6], A  src[6] + B  src[7])
    inc(dst[7], A  src[7] + B  src[8])
    inc(dst, stride)
    inc(src, stride)
    inc(i)

# Example 21: Backward access:

proc foo(b: ptr cint; n: cint): cint =
  var
    i: cint
    a: cint = 0

  i = n - 1
  while i >= 0:
    inc(a, b[i])
    dec(i)


  return a

# Example 22: Alignment hints:

proc foo(out1: ptr cint; in1: ptr cint; in2: ptr cint; n: cint) =
  var i: cint

  out1 = (
    out1
    32
    16)
  in1 = (
    in1
    32
    16)
  in2 = (
    in2
    32
    0)

  i = 0
  while i < n:
    out1[i] = in1[i]  in2[i]
    inc(i)

# Example 23: Widening shift:

proc foo(src: ptr cushort; dst: ptr cuint) =
  var i: cint

  i = 0
  while i < 256:
    inc(dst)[] = inc(src)[] shl 7
    inc(i)

# Example 24: Condition with mixed types:

const
  N = 1024

var
  a: array[N, cfloat]
  b: array[N, cfloat]

var c: array[N, cint]

proc foo(x: cshort; y: cshort) =
  var i: cint
  i = 0
  while i < N:
    c[i] = if a[i] < b[i]: x else: y
    inc(i)

# Example 25: Loop with bool:

const
  N = 1024

var
  a: array[N, cfloat]
  b: array[N, cfloat]
  c: array[N, cfloat]
  d: array[N, cfloat]

var j: array[N, cint]

proc foo() =
  var i: cint
  var
    x: bool
    y: bool
  i = 0
  while i < N:
    x = (a[i] < b[i])
    y = (c[i] < d[i])
    j[i] = x and y
    inc(i)
