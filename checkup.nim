template check(a, b: typed) =
   if a != b:
      echo("Failure: ", name)
      failure = true

template tryCheck(a, b: typed) =
   try:
      check(a, b)
   except:
      echo("Failure: ", name)
      failure = true

template tryCheckRaises(a: typed, e: type CatchableError) =
   try:
      discard a
      echo("Failure: ", name)
      failure = true
   except e:
      discard
   except:
      echo("Failure: ", name)
      failure = true

proc suite =
   var failure = false
   var name = "suite"
   echo(name, "...")

   name = "test1"
   echo(name, "...")
   block test1:
      let a = "Hello"
      let b = a & " World"
      check(a, b)

   name = "test2"
   echo(name, "...")
   block test2:
      var a = @[2]
      tryCheck(a.pop(), 2)

   name = "test3"
   echo(name, "...")
   block test3:
      var a: seq[int]
      tryCheckRaises(a.pop(), KeyError)

   if not failure:
      echo("Success: test")

suite()

suite("Test suite"):
   test("First test"):
      let a = "Hello"
      let b = a & " World"
      check(a, b)
   test("Second test"):
      var a = @[2]
      tryCheck(a.pop(), 2)
   test("Third test"):
      var a: seq[int]
      tryCheckRaises(a.pop(), KeyError)
