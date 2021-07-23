import std/isolation, threading/atomics
from typetraits import supportsCopyMem

proc raiseNilAccess() {.noinline.} =
  raise newException(NilAccessDefect, "dereferencing nil smart pointer")

template checkNotNil(p: typed) =
  when compileOption("boundChecks"):
    {.line.}:
      if p.isNil:
        raiseNilAccess()

type
  Deleter*[T] = proc (val: ptr Payload[T])
  Payload*[T] = object
    value*: T
    counter*: Atomic[int]
  SharedPtr*[T] = object
    ## Shared ownership reference counting pointer.
    deleter*: Deleter[T]
    val*: ptr Payload[T]

proc `=destroy`*[T](p: var SharedPtr[T]) =
  if p.val != nil:
    if p.val.counter.load(Consume) == 0:
      `=destroy`(p.val.value)
      if p.deleter != nil:
        p.deleter(p.val)
    else:
      atomicDec(p.val.counter)

proc defaultDel[T](val: ptr Payload[T]) =
  deallocShared(val)

proc `=copy`*[T](dest: var SharedPtr[T], src: SharedPtr[T]) =
  if src.val != nil:
    atomicInc(src.val.counter)
  if dest.val != nil:
    `=destroy`(dest)
  dest.val = src.val

proc newSharedPtr*[T](val: sink Isolated[T]): SharedPtr[T] {.nodestroy.} =
  ## Returns a shared pointer which shares
  ## ownership of the object by reference counting.
  result.val = cast[ptr Payload[T]](allocShared(sizeof(Payload[T])))
  int(result.val.counter) = 0
  result.val.value = extract val
  result.deleter = defaultDel

proc isNil*[T](p: SharedPtr[T]): bool {.inline.} =
  p.val == nil

proc `[]`*[T](p: SharedPtr[T]): var T {.inline.} =
  checkNotNil(p)
  p.val.value

when isMainModule:
  type
    Foo = object
      s: string

  proc `=destroy`(x: var Foo) =
    echo "destroying ", x.s
    `=destroy`(x.s)

  proc test =
    let x = newSharedPtr(isolate(Foo(s: "Hello World")))

  proc main =
    test()
    echo "exit"

  main()
