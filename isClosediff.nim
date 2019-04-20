# Use: --passC:'-S -fverbose-asm -masm=intel'

func isClose1(a, b: float; relTol = 1e-5, absTol = 1e-8): bool =
   let diff = abs(b - a)
   result = (diff <= abs(relTol * b) or diff <= abs(relTol * a)) or diff <= absTol

func isClose2(a, b: float; relTol = 1e-5, absTol = 1e-8): bool =
   result = abs(a - b) <= (absTol + relTol * abs(b))

func isClose3(a, b: float; relTol = 1e-5, absTol = 1e-8): bool =
   result = abs(a - b) <= (absTol + relTol * max(abs(a), abs(b)))

func isClose4(a, b: float; relTol = 1e-5, absTol = 1e-8): bool =
   result = abs(a - b) <= max(relTol * max(abs(a), abs(b)), absTol)

block test1:
   let a = 0.142253
   let b = 0.142219
   let rtol = 1e-4
   let atol = 2e-5
   doassert(not isClose1(a, b, rtol, atol))
   doassert(not isClose2(a, b, rtol, atol))
   doassert(not isClose3(a, b, rtol, atol))
   doassert(not isClose4(a, b, rtol, atol))

block test2:
   let a = 0.142253
   let b = 0.142219
   let rtol = 1e-4
   let atol = 1.9776e-05
   doassert(not isClose1(a, b, rtol, atol))
   doassert(not isClose2(a, b, rtol, atol))
   doassert(not isClose3(a, b, rtol, atol))
   doassert(not isClose4(a, b, rtol, atol))
