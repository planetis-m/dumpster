type
  ScopedFile = distinct File

proc `=destroy`(x: var ScopedFile) =
  close(File(x))

proc `=copy`*(dest: var ScopedFile; source: ScopedFile) {.error.}

proc main =
  var f = ScopedFile(open("temp.txt", fmWrite))
  File(f).write "Hello World\n"

main()
