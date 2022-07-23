import std/setutils

proc toSet*[T: SomeInteger; E](x: T, t: typedesc[E]): set[E] {.inline.} =
  if (x and cast[T](fullSet(E))) == x: result = cast[set[E]](x)
  else: raise newException(ValueError, $x & " can't be converted to " & $t)

when isMainModule:
  type
    Foo = enum
      bar, baz, baf

  let x = toSet(8, Foo)
  echo x
