const numPartials = 32 # initial partials array size, on stack

# Extend the partials array p[] by doubling its size.

proc fsum_realloc(p_ptr: ptr ptr float; n: int; ps: ptr float;
                  m_ptr: ptr int): cint =
   var v: pointer = nil
   var m: int = m_ptr[]

   m += m # double
   if n < m and
         m < (high(int) div sizeof(float)):
      var p: ptr float = p_ptr[]
      if p == ps:
         v = alloc(sizeof(float) * m)
      else:
         v = realloc(p, sizeof(float) * m)

   if v == nil:
      # size overflow or no memory
      # raise newException(OutOfMemError, "math.fsum partials")
      return 1

   p_ptr[] = cast[ptr float](v)
   m_ptr[] = m
   return 0

var ps: ptr float = cast[ptr float](alloc(numPartials))
var p: ptr float = ps

var n = 0
var m = numPartials
while true:
   assert 0 <= n and n <= m
   assert (m == numPartials and p == ps) or
      (m > numPartials and p != nil)

   n.inc
   if n >= m and fsumRealloc(addr(p), n, ps, addr(m)) > 0:
      quit("error")
   if n > 65: break

dealloc(ps)
