import std/isolation, threading/atomics

proc raiseNilAccess() {.noinline.} =
  raise newException(NilAccessDefect, "dereferencing nil smart pointer")

template checkNotNil(p: typed) =
  when compileOption("boundChecks"):
    {.line.}:
      if p.isNil:
        raiseNilAccess()

type
  OwnedPtr*[T] = object
    val: ptr tuple[value: T, counter: Atomic[int]]

  UnownedPtr*[T] = distinct OwnedPtr[T]

proc `=destroy`*[T](p: var OwnedPtr[T]) =
  mixin `=destroy`
  if p.val != nil:
    assert p.val.counter.load(Consume) == 0,
      "dangling unowned pointers exist!"
    `=destroy`(p.val[])
    deallocShared(p.val)

proc `=copy`*[T](dest: var OwnedPtr[T]; src: OwnedPtr[T]) {.
    error: "owned refs can only be moved".}

proc `=destroy`*[T](p: var UnownedPtr[T]) =
  if OwnedPtr[T](p).val != nil: atomicDec(OwnedPtr[T](p).val.counter)

proc `=copy`*[T](dest: var UnownedPtr[T]; src: UnownedPtr[T]) =
  # No need to check for self-assignments here.
  if OwnedPtr[T](src).val != nil: atomicInc(OwnedPtr[T](src).val.counter)
  if OwnedPtr[T](dest).val != nil: atomicDec(OwnedPtr[T](dest).val.counter)
  OwnedPtr[T](dest).val = OwnedPtr[T](src).val # raw pointer copy

proc `=sink`*[T](dest: var UnownedPtr[T], src: UnownedPtr[T]) {.
    error: "moves are not available for unowned refs".}

proc newOwnedPtr*[T](val: sink Isolated[T]): OwnedPtr[T] {.nodestroy.} =
  result.val = cast[typeof(result.val)](allocShared(sizeof(result.val[])))
  int(result.val.counter) = 0
  result.val.value = extract val

template newOwnedPtr*[T](val: T): OwnedPtr[T] =
  newOwnedPtr(isolate(val))

proc unown*[T](p: OwnedPtr[T]): UnownedPtr[T] =
  result = UnownedPtr[T](p)

proc dispose*[T](p: var UnownedPtr[T]) {.inline.} =
  `=destroy`(p)
  wasMoved(p)

proc isNil*[T](p: OwnedPtr[T]): bool {.inline.} =
  p.val == nil

proc isNil*[T](p: UnownedPtr[T]): bool {.inline.} =
  OwnedPtr[T](p).val == nil

proc `[]`*[T](p: OwnedPtr[T]): var T {.inline.} =
  checkNotNil(p)
  p.val.value

proc `[]`*[T](p: UnownedPtr[T]): var T {.inline.} =
  checkNotNil(p)
  OwnedPtr[T](p).val.value

proc `$`*[T](p: OwnedPtr[T]): string {.inline.} =
  if p.val == nil: "nil"
  else: "(val: " & $p.val.value & ")"

proc `$`*[T](p: UnownedPtr[T]): string {.inline.} =
  $OwnedPtr[T](p)

when isMainModule:
  # https://nim-lang.org/araq/ownedrefs.html
  type
    Node = object
      data: int

  var x = newOwnedPtr(Node(data: 3))
  var dangling = unown x
  assert dangling[].data == 3
  dispose dangling
  # reassignment causes the memory of what `x` points to to be freed:
  x = newOwnedPtr(Node(data: 4))
  # accessing 'dangling' here is invalid as it is nil.
  # at scope exit the memory of what `x` points to is freed
  #assert dangling[].data == 3
