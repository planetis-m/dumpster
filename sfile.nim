import std/syncio

type
  ScopedFile* = distinct File

proc `=copy`*(dest: var ScopedFile; source: ScopedFile) {.error.}
proc `=destroy`*(x: ScopedFile) =
  close(File(x))

proc main =
  var f = ScopedFile(open("temp.txt", fmWrite))
  File(f).write "Hello World\n"

main()
