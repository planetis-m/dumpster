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
    deleter*: Deleter[T]
  SharedPtrImpl*[T] = object
    ## Shared ownership reference counting pointer.
    val*: ptr Payload[T]
  SharedPtr*[T] = distinct SharedPtrImpl[T]

proc `=destroy`*[T](p: var SharedPtrImpl[T]) =
  if p.val != nil:
    if p.val.counter.load(Consume) == 0:
      `=destroy`(p.val.value)
      if p.val.deleter != nil:
        p.val.deleter(p.val)
    else:
      atomicDec(p.val.counter)

proc defaultDel[T](val: ptr Payload[T]) =
  deallocShared(val)

proc `=copy`*[T](dest: var SharedPtrImpl[T], src: SharedPtrImpl[T]) =
  if src.val != nil:
    atomicInc(src.val.counter)
  if dest.val != nil:
    `=destroy`(dest)
  dest.val = src.val

proc newSharedPtr*[T](val: sink Isolated[T]): SharedPtr[T] {.nodestroy.} =
  ## Returns a shared pointer which shares
  ## ownership of the object by reference counting.
  SharedPtrImpl[T](result).val = cast[ptr Payload[T]](allocShared(sizeof(Payload[T])))
  int(SharedPtrImpl[T](result).val.counter) = 0
  SharedPtrImpl[T](result).val.value = extract val
  SharedPtrImpl[T](result).val.deleter = defaultDel

proc isNil*[T](p: SharedPtr[T]): bool {.inline.} =
  SharedPtrImpl[T](p).val == nil

proc `[]`*[T](p: SharedPtr[T]): var T {.inline.} =
  checkNotNil(p)
  SharedPtrImpl[T](p).val.value

when isMainModule:
  type
    Foo = object
      s: string

  proc `=destroy`(x: var Foo) =
    echo "destroying ", x.s
    `=destroy`(x.s)

  proc test =
    let x = newSharedPtr(isolate(Foo(s: "Hello World")))
    echo x[].s

  proc main =
    test()
    echo "exit"

  main()
