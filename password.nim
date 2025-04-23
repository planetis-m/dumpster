import std/sysrand, strutils
let a = urandom(100)
var s = ""
for i in a:
  if char(i) in Letters + PunctuationChars + Digits:
    s.add char(i)
echo s
