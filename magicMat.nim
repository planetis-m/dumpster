import std/[assertions, formatfloat]

proc magicSquare(n: int): seq[float] =
   assert n != 2, "Magic square of order 2 cannot be constructed"
   template `[]`(s: seq; a, b: int): untyped = s[a * n + b]
   template `[]=`(s: seq; a, b: int; val: untyped) = s[a * n + b] = float(val)
   var m = newSeq[float](n * n)
   # Odd order
   if n mod 2 == 1:
      let a = (n + 1) div 2
      let b = n + 1
      for j in 0 ..< n:
         for i in 0 ..< n:
            m[i, j] = n * ((i + j + a) mod n) + ((i + 2 * j + b) mod n) + 1
   # Doubly Even Order
   elif n div 4 == 0:
      for j in 0 ..< n:
         for i in 0 ..< n:
            if (i + 1) div 2 mod 2 == (j + 1) div 2 mod 2:
               m[i, j] = n * n - n * i - j
            else:
               m[i, j] = n * i + j + 1
   # Singly Even Order
   else:
      let p = n div 2
      let k = (n - 2) div 4
      let a = magicSquare(p)
      for j in 0 ..< p:
         for i in 0 ..< p:
            let aij = a[i, j]
            m[i, j] = aij
            m[i, j + p] = aij + float(2 * p * p)
            m[i + p, j] = aij + float(3 * p * p)
            m[i + p, j + p] = aij + float(p * p)
      for i in 0 ..< p:
         for j in 0 ..< k:
            swap(m[i, j], m[i + p, j])
         for j in n - k + 1 ..< n:
            swap(m[i, j], m[i + p, j])
      swap(m[k, 0], m[k + p, 0])
      swap(m[k, k], m[k + p, k])
   result = m

proc display(data: seq[float]; n: int) =
   for i in 0 ..< n:
      var line = ""
      let offset = i * n
      for j in 0 ..< n:
         if line.len > 0: line.add(", ")
         line.addFloat(data[offset + j])
      echo line

let m5 = magicSquare(5)
display(m5, 5)
