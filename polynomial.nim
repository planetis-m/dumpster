## Program: Single header polynomial library for the 2nd Assignment
## Programmer: Antonis Geralis
## Date Created: 12/4/22
type
  Polynomial* = object
    data: seq[int] ## Storage for coefficients

proc degree*(p: Polynomial): int {.inline.} = p.data.len - 1

proc poly*(degree: int): Polynomial =
  result = Polynomial(data: newSeq[int](degree + 1)) # +1 for constant

proc parseNum(x: string, number: var int, start = 0): int =
  var i = start
  if i < x.len and x[i] in {'0'..'9'}:
    while true:
      number = number * 10 + x[i].int - '0'.int
      inc i
      if i >= x.len or x[i] notin {'0'..'9'}: break
  result = i - start

proc poly*(x: string): Polynomial =
  var pos = 0
  while pos < x.len:
    var sign = 1
    if pos < x.len and x[pos] == '-':
      sign = -1
      inc pos
    elif pos < x.len and x[pos] == '+':
      inc pos
    var number = 0
    var temp = parseNum(x, number, pos)
    pos += temp
    if pos < x.len and x[pos] == 'x':
      if temp == 0: number = 1
      inc pos
    number = number * sign
    var exponent = 0
    temp = parseNum(x, exponent, pos)
    if temp == 0 and (pos - 1 < x.len and x[pos - 1] == 'x'):
      exponent = 1
    pos += temp
    if exponent > result.degree:
      grow(result.data, exponent + 1, 0)
    result.data[exponent] += number # allow multiple terms of the same degree

proc `[]`*(p: Polynomial, idx: int): int {.inline.} =
  result = if idx >= 0 and idx < p.data.len: p.data[idx] else: 0

proc `+`*(p1, p2: Polynomial): Polynomial =
  result = poly(max(p1.degree, p2.degree))
  for i in 0 .. high(result.data):
    result.data[i] = p1[i] + p2[i]

proc `$`*(p: Polynomial): string =
  result = ""
  var first = true
  for i in countdown(p.degree, 0):
    let a = p[i]
    if a != 0:
      if not first and a > 0:
        result.add '+'
      else:
        first = false
      if a == -1 and i != 0:
        result.add '-'
      elif a != 1 or i == 0:
        result.addInt a
      if i > 0:
        result.add 'x'
      if i >= 2:
        result.addInt i
  if first:
    result = "0"

when isMainModule:
  # Polynomial constructors
  block empty1:
    let poly = poly("")
    assert -1 == poly.degree
  block degree0:
    let poly = poly("-2")
    assert 0 == poly.degree
    assert -2 == poly[0]
  block degree1:
    let poly = poly("x-2")
    assert 1 == poly.degree
    assert -2 == poly[0]
    assert 1 == poly[1]
  block degree4:
    let poly = poly("3x4+5x3+x")
    assert 4 == poly.degree
    let data = [0, 1, 0, 5, 3]
    for i in 0 ..< 5:
      assert data[i] == poly[i]
  block degree5:
    let poly = poly("2x5-5x4-x+2")
    assert 5 == poly.degree
    let data = [2, -1, 0, 0, -5, 2]
    for i in 0 ..< 6:
      assert data[i] == poly[i]
  block degree10:
    let poly = poly("-x10")
    assert 10 == poly.degree
    assert -1 == poly[10]
    for i in 0 ..< 10:
      assert 0 == poly[i]
  # Polynomial summation
  block empty2:
    var poly1, poly2: Polynomial
    assert -1 == poly2.degree
    let poly3 = poly1 + poly2
    assert -1 == poly3.degree
  block case1:
    let poly1 = poly("3x4+5x3+x")
    let poly2 = poly("2x5-5x4-x+2")
    let poly3 = poly1 + poly2
    assert 5 == poly3.degree
    let data = [2, 0, 0, 5, -2, 2]
    for i in 0 ..< 6:
      assert data[i] == poly3[i]
