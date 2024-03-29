import std/isolation

type
  StrPayloadBase = object
    cap, counter: int

  StrPayload = object
    cap, counter: int
    data: UncheckedArray[char]

  String* = object
    len: int
    p: ptr StrPayload ## can be nil if len == 0.

template contentSize(cap): int = cap + 1 + sizeof(StrPayloadBase)

template frees(s) =
  when compileOption("threads"):
    deallocShared(s.p)
  else:
    dealloc(s.p)

proc `=destroy`*(x: var String) =
  if x.p != nil:
    if x.p.counter == 0:
      frees(x)
    else:
      dec x.p.counter

proc `=copy`*(a: var String, b: String) =
  if b.p != nil:
    inc b.p.counter
  if a.p != nil:
    `=destroy`(a)
  a.p = b.p
  a.len = b.len

proc deepCopy*(y: String): String =
  result.len = y.len
  if y.p != nil:
    when compileOption("threads"):
      result.p = cast[ptr StrPayload](allocShared0(contentSize(result.len)))
    else:
      result.p = cast[ptr StrPayload](alloc0(contentSize(result.len)))
    result.p.cap = result.len
    result.p.counter = 0
    if result.len > 0:
      # also copy the \0 terminator:
      copyMem(unsafeAddr result.p.data[0], unsafeAddr y.p.data[0], result.len+1)

proc resize(old: int): int {.inline.} =
  if old <= 0: result = 4
  elif old < 65536: result = old * 2
  else: result = old * 3 div 2 # for large arrays * 3/2 is better

proc prepareAdd(s: var String; addLen: int) =
  let newLen = s.len + addLen
  # copy the data iff there is more than a reference or its a literal
  if s.p == nil or s.p.counter != 0:
    let oldP = s.p # can be nil
    if s.p != nil and s.p.counter > 0: dec s.p.counter
    # can't mutate a literal, so we need a fresh copy here:
    when compileOption("threads"):
      s.p = cast[ptr StrPayload](allocShared0(contentSize(newLen)))
    else:
      s.p = cast[ptr StrPayload](alloc0(contentSize(newLen)))
    s.p.cap = newLen
    s.p.counter = 0
    if s.len > 0:
      # we are about to append, so there is no need to copy the \0 terminator:
      copyMem(unsafeAddr s.p.data[0], unsafeAddr oldP.data[0], min(s.len, newLen))
  else:
    let oldCap = s.p.cap
    if newLen > oldCap:
      let newCap = max(newLen, resize(oldCap))
      when compileOption("threads"):
        s.p = cast[ptr StrPayload](reallocShared0(s.p, contentSize(oldCap), contentSize(newCap)))
      else:
        s.p = cast[ptr StrPayload](realloc0(s.p, contentSize(oldCap), contentSize(newCap)))
      s.p.cap = newCap

proc add*(s: var String; c: char) {.inline.} =
  prepareAdd(s, 1)
  s.p.data[s.len] = c
  s.p.data[s.len+1] = '\0'
  inc s.len

proc add*(dest: var String; src: String) {.inline.} =
  if src.len > 0:
    prepareAdd(dest, src.len)
    # also copy the \0 terminator:
    copyMem(unsafeAddr dest.p.data[dest.len], unsafeAddr src.p.data[0], src.len+1)
    inc dest.len, src.len

proc cstrToStr(str: cstring, len: int): String =
  if len <= 0:
    result = String(len: 0, p: nil)
  else:
    when compileOption("threads"):
      let p = cast[ptr StrPayload](allocShared0(contentSize(len)))
    else:
      let p = cast[ptr StrPayload](alloc0(contentSize(len)))
    p.cap = len
    p.counter = 0
    if len > 0:
      copyMem(unsafeAddr p.data[0], str, len+1)
    result = String(len: len, p: p)

proc toStr*(str: cstring): String {.inline.} =
  if str == nil: cstrToStr(str, 0)
  else: cstrToStr(str, str.len)

proc toStr*(str: string): String {.inline.} =
  cstrToStr(str.cstring, str.len)

proc toCStr*(s: String): cstring {.inline.} =
  if s.len == 0: result = cstring""
  else: result = cstring(unsafeAddr s.p.data)

proc initStringOfCap*(space: int): String =
  # this is also 'system.newStringOfCap'.
  if space <= 0:
    result = String(len: 0, p: nil)
  else:
    when compileOption("threads"):
      let p = cast[ptr StrPayload](allocShared0(contentSize(space)))
    else:
      let p = cast[ptr StrPayload](alloc0(contentSize(space)))
    p.cap = space
    p.counter = 0
    result = String(len: 0, p: p)

proc initString*(len: int): String =
  if len <= 0:
    result = String(len: 0, p: nil)
  else:
    when compileOption("threads"):
      let p = cast[ptr StrPayload](allocShared0(contentSize(len)))
    else:
      let p = cast[ptr StrPayload](alloc0(contentSize(len)))
    p.cap = len
    p.counter = 0
    result = String(len: len, p: p)

proc setLen*(s: var String, newLen: int) =
  if newLen == 0:
    discard "do not free the buffer here, pattern 's.setLen 0' is common for avoiding allocations"
  else:
    if newLen > s.len or s.p == nil:
      prepareAdd(s, newLen - s.len)
    s.p.data[newLen] = '\0'
  s.len = newLen

proc len*(s: String): int {.inline.} = s.len

proc isolate*(value: sink String): Isolated[String] {.nodestroy.} =
  # Ensure unique
  if value.p == nil or value.p.counter <= 0:
    result = unsafeIsolate value
  else:
    result = unsafeIsolate deepCopy(value)

# Comparisons
proc eqStrings*(a, b: String): bool =
  result = false
  if a.len == b.len:
    if a.len == 0: result = true
    else: result = equalMem(unsafeAddr a.p.data[0], unsafeAddr b.p.data[0], a.len)

proc `==`*(a, b: String): bool {.inline.} = eqStrings(a, b)

proc cmpStrings*(a, b: String): int =
  let minLen = min(a.len, b.len)
  if minLen > 0:
    result = cmpMem(unsafeAddr a.p.data[0], unsafeAddr b.p.data[0], minLen)
    if result == 0:
      result = a.len - b.len
  else:
    result = a.len - b.len

proc `<=`*(a, b: String): bool {.inline.} = cmpStrings(a, b) <= 0
proc `<`*(a, b: String): bool {.inline.} = cmpStrings(a, b) < 0

proc raiseIndexDefect() {.noinline.} =
  raise newException(IndexDefect, "index out of bounds")

template checkBounds(cond: untyped) =
  when compileOption("boundChecks"):
    {.line.}:
      if not cond:
        raiseIndexDefect()

proc `[]`*(x: String; i: int): char {.inline.} =
  checkBounds(i >= 0 and i <= x.len)
  x.p.data[i]

proc `[]=`*(x: String; i: int; val: char) {.inline.} =
  checkBounds(i >= 0 and i <= x.len)
  x.p.data[i] = val
