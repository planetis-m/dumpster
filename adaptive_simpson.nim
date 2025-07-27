import std/[syncio, math]

type Func = proc(x: float): float

proc integrate(f: Func; a, b: float; eps = 1e-8; maxDepth = 20): float =
  proc adsimp(f: Func; a, b, eps, s, fa, fb, fc: float; depth: int): float =
    let
      c = (a + b)/2
      h = b - a
      d = (a + c)/2
      e = (c + b)/2
      fd = f(d)
      fe = f(e)
      sLeft = (h/12)*(fa + 4*fd + fc)
      sRight = (h/12)*(fc + 4*fe + fb)
      s2 = sLeft + sRight
    if depth <= 0 or abs(s2 - s) <= 15*eps: # magic 15 comes from error analysis
      return s2 + (s2 - s)/15
    result = adsimp(f, a, c, eps/2, sLeft, fa, fc, fd, depth-1) +
             adsimp(f, c, b, eps/2, sRight, fc, fb, fe, depth-1)
  let
    c = (a + b)/2
    h = b - a
    fa = f(a)
    fb = f(b)
    fc = f(c)
    s = (h/6)*(fa + 4*fc + fb)
  result = adsimp(f, a, b, eps, s, fa, fb, fc, maxDepth)

let i = integrate(sin, 0, 2, 0.001, 100)
echo("I = ", i)
