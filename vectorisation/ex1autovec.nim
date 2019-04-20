# Example 1:
{.passC: "-march=native -ffast-math -fno-unroll-loops -fopt-info -fverbose-asm -masm=intel -S".}
var
  a: array[1024, cint]
  b: array[1024, cint]
  c: array[1024, cint]

proc foo(n: cint) =
  var i: cint

  i = 0
  while i < n:
    a[i] = b[i] + c[i]
    inc(i)

foo(a.len)
