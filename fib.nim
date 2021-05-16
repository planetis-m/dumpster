# https://www.youtube.com/watch?v=QIHy8pXbneI
# https://sahandsaba.com/five-ways-to-calculate-fibonacci-numbers-with-python-code.html
import int128

type
  IntMat2 = array[4, Int128]

proc `*`(x, y: IntMat2): IntMat2 =
  result = [x[0] * y[0] + x[1] * y[2], x[0] * y[1] + x[1] * y[3],
            x[2] * y[0] + x[3] * y[2], x[2] * y[1] + x[3] * y[3]]

proc power(x: IntMat2, n: int): IntMat2 =
  if n == 0: result = [One, Zero, Zero, One]
  else:
    var
      n = n
      x = x
    while (n and 1) == 0:
      n = n shr 1
      x = x * x
    result = x
    n = n shr 1
    while n != 0:
      x = x * x
      if (n and 1) != 0: result = result * x
      n = n shr 1

proc fib(n: Natural): Int128 =
  if n == 0:
    result = Zero
  else:
    result = power([One, One, One, Zero], n - 1)[0]

proc `'i128`(n: string): Int128 = parseDecimalInt128(n)

assert fib(46) == 1836311903.toInt128
assert fib(92) == 7540113804746346429.toInt128
assert fib(184) == 127127879743834334146972278486287885163'i128
