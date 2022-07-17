# nim c --cc=clang --clang.exe=afl-clang --clang.linkerexe=afl-clang --mm:orc --panics:on
# -d:useMalloc -d:release -g -d:noSignalHandler -d:afl tafl
# afl-fuzz -i - -o results -- ./tafl
import posix, std/syncio

template suicide() =
  discard kill(getpid(), SIGSEGV)

template readStdIn(): untyped =
  cast[seq[byte]](stdin.readAll)

proc fuzzMe(s: string; a, b, c: int32) =
  if a == 0xdeadbeef'i32 and b == 0x11111111'i32 and c == 0x22222222'i32:
    if s.len == 100: echo "PANIC!"; suicide()

proc main =
  let payload = readStdin()
  var pos = 0
  if payload.len < sizeof(int32) * 3 + 100: return
  let s = newString(100)
  copyMem(cstring(s), addr payload[pos], s.len)
  inc pos, 100
  var a, b, c: int32
  copyMem(addr a, addr payload[pos], sizeof(a))
  inc pos, sizeof(a)
  copyMem(addr b, addr payload[pos], sizeof(b))
  inc pos, sizeof(b)
  copyMem(addr c, addr payload[pos], sizeof(c))
  fuzzMe(s, a, b, c)

main()
