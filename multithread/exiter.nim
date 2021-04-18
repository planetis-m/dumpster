import std / locks

var lock: Lock

iterator items*(a: seq[int]): int =
  var i = 0
  while i < len(a):
    withLock lock:
      yield a[i]
    inc(i)

proc main =
  initLock lock
  var a = newSeq[int](5)
  for x in items(a):
    echo x
  deinitLock lock

main()
