import benchitexp

type
   S = object
      u: uint64
      d: float64
      i: int32
      f: float32

   Data = object
      vu: seq[uint64]
      vd: seq[float64]
      vi: seq[int32]
      vf: seq[float32]

proc test1(s1, s2: S): int =
   s1.i + s2.i

proc test2(data: Data, ind1, ind2: int): int =
   data.vi[ind1] + data.vi[ind2]

const
   N = 30000
   R = 10

warmup()

var v = newSeq[S](N)
var data: Data
data.vu = newSeq[uint64](N)
data.vd = newSeq[float64](N)
data.vi = newSeq[int32](N)
data.vf = newSeq[float32](N)

benchIt("test #1", int, R):
   for a in 0 .. v.len - 2:
      for b in a + 1 ..< v.len:
         it += test1(v[a], v[b])

benchIt("test #2", int, R):
   for a in 0 .. data.vi.len - 2:
      for b in a + 1 ..< data.vi.len:
         it += test2(data, a, b)
