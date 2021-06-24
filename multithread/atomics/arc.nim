type
  Arc*[T] = object
    ## A thread-safe reference-counting pointer. 'Arc' stands for 'Atomically Reference Counted'.
    val: ptr tuple[value: T, atomicCounter: int]

proc `=destroy`*[T](p: var Arc[T]) =
  mixin `=destroy`
  if p.val != nil:
    if atomicLoadN(addr p.val[].atomicCounter, ATOMIC_CONSUME) == 0:
      `=destroy`(p.val[])
      deallocShared(p.val)
    else:
      discard atomicDec(p.val[].atomicCounter)

proc `=`*[T](dest: var Arc[T], src: Arc[T]) =
  if src.val != nil:
    discard atomicInc(src.val[].atomicCounter)
  if dest.val != nil:
    `=destroy`(dest)
  dest.val = src.val

proc newArc*[T](val: sink T): Arc[T] {.nodestroy.} =
  result.val = cast[typeof(result.val)](allocShared(sizeof(result.val[])))
  result.val.atomicCounter = 0
  result.val.value = val

proc isNil*[T](p: Arc[T]): bool {.inline.} =
  p.val == nil

proc `[]`*[T](p: Arc[T]): var T {.inline.} =
  when compileOption("boundChecks"):
    doAssert(p.val != nil, "deferencing nil shared pointer")
  p.val.value

proc `$`*[T](p: Arc[T]): string {.inline.} =
  if p.val == nil: "Arc[" & $T & "](nil)"
  else: "Arc[" & $T & "](" & $p.val.value & ")"


block: # Arc[T] test
  var a1: Arc[float]
  let a2 = newArc(0)

  assert $a1 == "Arc[float](nil)"
  assert a1.isNil == true
  assert $a2 == "Arc[int](0)"
  assert a2.isNil == false
  assert a2[] == 0
