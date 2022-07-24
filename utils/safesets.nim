import std/setutils

proc toSet*[T: SomeUnsignedInt; E](x: T, t: typedesc[E]): set[E] {.inline.} =
  if (x and cast[T](fullSet(E))) == x: result = cast[set[E]](x)
  else: raise newException(ValueError, $x & " can't be converted to a set of " & $t)

when isMainModule:
  type
    Foo = enum
      bar, baz = 7, baf = 15

  let x = toSet(128, Foo)
  echo x
