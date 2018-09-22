# https://codereview.stackexchange.com/q/187106
const
   Separator = "###"

type
   IllegalState = object of Exception

   MatchReplacePair = object
      match, replace: string

   MatcherState = enum
      SeparatorExpected, MatchExpected, ReplaceExpected

proc readRegexesFromFile(file: File): seq[MatchReplacePair] =
   let fs = newFileStream(file)
   try:
      var line = newString(80)
      var state = SeparatorExpected
      var pair = MatchReplacePair()
      while fs.readLine(line):
         case state
         of SeparatorExpected:
            if line.startsWith(Separator):
               state = MatchExpected
            else:
               raise newException(IllegalState, "Separator expected, but not found.")
         of MatchExpected:
            pair.match = line
            state = ReplaceExpected
         of ReplaceExpected:
            pair.replace = line
            result.add(pair)
            state = SeparatorExpected
         else:
            raise newException(IllegalState,
               "Unknown state. Please contact the developer.")
      if state != SeparatorExpected:
         raise newException(IllegalState,
            "Incorrect file contents: make sure the file ends with a \"replace\" block.")
   except IOError as e:
      log.write(e.getStackTrace)

discard """
### A comment
match
replace
"""
