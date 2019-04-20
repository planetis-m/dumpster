# http://www.cs.toronto.edu/~heap/270F02/node64.html
proc mult(x: openArray[(string, int, int)]): tuple[cost: int, order: string, rows, cols: int] =
   # return the linear index of row, column
   template `[]`(s: seq; a, b: int): untyped = s[a * n + b]
   template `[]=`(s: seq; a, b: int; val: untyped) = s[a * n + b] = val
   let n = len(x)
   var aux = newSeq[(int, string, int, int)](n * n)
   for i in 0 ..< n:
      let (tname, trow, tcol) = x[i]
      # single matrix chain has zero cost
      aux[i, i] = (0, tname, trow, tcol)
   # i: length of subchain
   for i in 1 ..< n:
      # j: starting index of subchain
      for j in 0 ..< n - i:
         var best = high(int)
         # k: splitting point of subchain
         for k in j ..< j + i:
            # multiply subchains at splitting point
            let
               (lcost, lname, lrow, lcol) = aux[j, k]
               (rcost, rname, rrow, rcol) = aux[k + 1, j + i]
               cost = lcost + rcost + lrow * lcol * rcol
               tvar = "(" & lname & rname & ")"
            # pick the best one
            if cost < best:
               best = cost
               aux[j, j + i] = (cost, tvar, lrow, rcol)
   aux[0, n - 1]


echo mult([("A", 10, 20), ("B", 20, 30), ("C", 30, 40)])
# (cost: 18000, order: ((AB)C), rows: 10, cols: 40)
echo mult([("A", 10, 5), ("B", 5, 1), ("C", 1, 5), ("D", 5, 10), ("E", 10, 1)])
# (cost: 110, order: (A(B(C(DE)))), rows: 10, cols: 1)
