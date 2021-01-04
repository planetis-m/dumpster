type
   CStringArray* = object
      len*: int
      impl*: cstringArray

proc `=destroy`*(cstrs: var CStringArray) =
   if cstrs.impl != nil:
      for i in 0 ..< cstrs.len:
         deallocShared(cstrs.impl[i])
      deallocShared(cstrs.impl)

proc `=`*(dest: var CStringArray, source: CStringArray) =
   if dest.impl != source.impl:
      `=destroy`(dest)
      wasMoved(dest)
      dest.impl = cast[cstringArray](allocShared(sizeof(cstring) * source.len))
      for i in 0 ..< source.len:
         let cstrLen = source.impl[i].len + 1
         dest.impl[i] = cast[cstring](allocShared(cstrLen))
         copyMem(dest.impl[i], addr source.impl[i][0], cstrLen)

proc newCStringArray*(len: Natural): CStringArray =
   let impl = cast[cstringArray](allocShared0(sizeof(cstring) * len))
   result = CStringArray(len: len, impl: impl)

proc newCStringArray*(a: openArray[cstring]): CStringArray =
   let impl = cast[cstringArray](allocShared(a.len * sizeof(cstring)))
   let x = cast[ptr UncheckedArray[cstring]](a)
   for i in 0 ..< a.len:
      let cstrLen = x[i].len + 1
      impl[i] = cast[cstring](alloc(cstrLen))
      copyMem(impl[i], addr x[i][0], cstrLen)
   result = CStringArray(len: a.len, impl: impl)
