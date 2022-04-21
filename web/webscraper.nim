# https://forum.nim-lang.org/t/8882#58094
import std/[os, streams, parsexml, strutils]

proc main =
  let params = commandLineParams()
  if "--help" in params or "-h" in params or params.len != 1:
    quit("Usage: htmlrefs filename[.html]")

  let filename = addFileExt(params[0], ".html")
  let s = newFileStream(filename, fmRead)
  if s == nil: quit("cannot open the file " & filename)
  var x: XmlParser
  open(x, s, filename)
  while true:
    next(x)
    if x.kind == xmlEof: break
    if x.kind == xmlAttribute and cmpIgnoreCase(x.attrKey, "href") == 0:
      echo "found a link: ", x.attrValue
  x.close()

main()
