## This template provides syntactic sugar and
## emulates/substitutes an optional type

type
   SomePointer = ref | ptr | pointer | proc

template `?=`*(value, call: untyped): bool =
   ## Operator that unwraps the "optional" value that `call` returns, into `value`.
   ## This can be `(bool, T)` or any pointer type.
   ## Used only as an if condition
   when call is SomePointer:
      (let value = call; system.`!=`(value, nil))
   else:
      (let (isSome, value) = call; isSome)

when isMainModule:
   # proc that returns (bool, T)
   proc test1(): (bool, int) =
      if true: (true, 2) # some value
      else: (false, 0) # none

   if a ?= test1():
      echo "value: ", a

   type
      Foo = ref object
         value: int
   # proc that returns ref
   proc test2(): Foo = discard

   if a ?= test2():
      echo "fail"
