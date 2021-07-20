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

#[proc `=deepCopy`*(a: var String, b: String) =
  if a.p != nil:
    `=destroy`(a)
  a.len = b.len
  if b.p != nil:
    when compileOption("threads"):
      a.p = cast[ptr StrPayload](allocShared0(contentSize(a.len)))
    else:
      a.p = cast[ptr StrPayload](alloc0(contentSize(a.len)))
    a.p.cap = a.len
    a.p.counter = 0
    if a.len > 0:
      # also copy the \0 terminator:
      copyMem(unsafeAddr a.p.data[0], unsafeAddr b.p.data[0], a.len+1)]#

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

proc add(dest: var String; src: String) {.inline.} =
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
      # we are about to append, so there is no need to copy the \0 terminator:
      copyMem(unsafeAddr p.data[0], str, len)
    result = String(len: len, p: p)

proc toStr*(str: cstring): String =
  if str == nil: cstrToStr(str, 0)
  else: cstrToStr(str, str.len)

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

func isolate*(value: String): Isolated[String] =
  var temp: String
  temp.len = value.len
  if value.p != nil:
    when compileOption("threads"):
      temp.p = cast[ptr StrPayload](allocShared0(contentSize(temp.len)))
    else:
      temp.p = cast[ptr StrPayload](alloc0(contentSize(temp.len)))
    temp.p.cap = temp.len
    temp.p.counter = 0
    if temp.len > 0:
      # also copy the \0 terminator:
      copyMem(unsafeAddr temp.p.data[0], unsafeAddr value.p.data[0], temp.len+1)
  result = unsafeIsolate temp

when isMainModule:
  proc main =
    block:
      var a = initStringOfCap(10)
      a.add 'h'
      var b = a
      b.add a
      b.add 'w'
      echo a.toCStr # prevent sink
      echo b.toCStr
    block:
      var a: String
      a.add 'h'
      a.add 'e'
      echo a.toCStr
    block:
      var a: Isolated[String]
      var b: String
      b.add 'w'
      b.add 'o'
      a = isolate b
      #b.add 'r'
      let c = extract a
      assert cast[ByteAddress](c.p) != cast[ByteAddress](b.p)
      echo c.toCStr

  main()
