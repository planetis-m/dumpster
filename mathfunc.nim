import math

proc isClose(a, b: float; relTol = 1e-09, absTol = 0.0): bool =
   # sanity check on the inputs
   assert(relTol >= 0.0 and absTol >= 0.0, "tolerances must be non-negative")
   if a == b:
      # short circuit exact equality -- needed to catch two infinities of
      # the same sign. And perhaps speeds things up a bit sometimes.
      return true
   # This catches the case of two infinities of opposite sign, or
   # one infinity and one finite number. Two infinities of opposite
   # sign would otherwise have an infinite relative tolerance.
   # Two infinities of the same sign are caught by the equality check
   # above.
   template isInfinity(x): bool = classify(x) in {fcInf, fcNegInf}
   if a.isInfinity or b.isInfinity:
      return false
   # now do the regular computation
   # this is essentially the "weak" test from the Boost library
   let diff = abs(b - a)
   result = (diff <= abs(relTol * b) or diff <= abs(relTol * a)) or diff <= absTol

when isMainModule:
   assert isClose(1.0, 1.001) == false
   assert isClose(1.0, 1.0000000001) == true
   assert isClose(1.0, 2.0000000001) == false
   assert isClose(Inf, Inf) == true
   assert isClose(-Inf, -Inf) == true
   assert isClose(Inf, -Inf) == false
   assert isClose(Inf, NaN) == false
   assert isClose(NaN, NaN) == false
