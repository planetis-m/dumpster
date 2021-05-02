when not compileOption("threads"):
  {.error: "This module requires --threads:on compilation flag".}

type
  Payload[T] = tuple[value: T, atomicCounter: int]
  Owned*[T] = object
    val: ptr Payload[T]

  Unowned*[T] = object
    val: ptr Payload[T]

proc `=destroy`*[T](p: var Owned[T]) =
  mixin `=destroy`
  if p.val != nil:
    assert atomicLoadN(addr p.val[].atomicCounter, AtomicConsume) == 0,
      "dangling unowned pointers exist!"
    `=destroy`(p.val[])
    deallocShared(p.val)

proc `=copy`*[T](dest: var Owned[T]; src: Owned[T]) {.
    error: "owned refs can only be moved".}

proc `=destroy`*[T](p: var Unowned[T]) =
  if p.val != nil: atomicDec(p.val[].atomicCounter)

proc `=copy`*[T](dest: var Unowned[T]; src: Unowned[T]) =
  # No need to check for self-assignments here.
  if src.val != nil: atomicInc(src.val[].atomicCounter)
  if dest.val != nil: atomicDec(dest.val[].atomicCounter)
  dest.val = src.val # raw pointer copy

proc `=sink`*[T](dest: var Unowned[T], src: Unowned[T]) {.
    error: "moves are not available for unowned refs".}

proc newOwned*[T](val: sink T): Owned[T] {.nodestroy.} =
  result.val = cast[typeof(result.val)](allocShared(sizeof(result.val[])))
  result.val.atomicCounter = 0
  result.val.value = val

proc unown*[T](p: Owned[T]): Unowned[T] =
  if p.val != nil: atomicInc(p.val[].atomicCounter)
  result.val = p.val

proc dispose*[T](p: var Unowned[T]) {.inline.} =
  `=destroy`(p)
  wasMoved(p)

proc isNil*[T](p: Owned[T]): bool {.inline.} =
  p.val == nil

proc isNil*[T](p: Unowned[T]): bool {.inline.} =
  p.val == nil

proc `[]`*[T](p: Owned[T]): var T {.inline.} =
  when compileOption("boundChecks"):
    doAssert(p.val != nil, "deferencing nil shared pointer")
  p.val.value

proc `[]`*[T](p: Unowned[T]): var T {.inline.} =
  when compileOption("boundChecks"):
    doAssert(p.val != nil, "deferencing nil shared pointer")
  p.val.value

proc `$`*[T](p: Owned[T]): string {.inline.} =
  if p.val == nil: "Owned[" & $T & "](nil)"
  else: "Owned[" & $T & "](" & $p.val.value & ")"

proc `$`*[T](p: Unowned[T]): string {.inline.} =
  if p.val == nil: "Unowned[" & $T & "](nil)"
  else: "Unowned[" & $T & "](" & $p.val.value & ")"

when isMainModule:
  # https://nim-lang.org/araq/ownedrefs.html
  type
    Node = object
      data: int

  var x = newOwned(Node(data: 3))
  let dangling = unown x
  assert dangling[].data == 3
  dispose dangling
  # reassignment causes the memory of what ``x`` points to to be freed:
  x = newOwned(Node(data: 4))
  # accessing 'dangling' here is invalid as it is nil.
  # at scope exit the memory of what ``x`` points to is freed
  #assert dangling[].data == 3
