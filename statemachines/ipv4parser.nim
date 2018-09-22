proc find(text: string): bool =
   # [0-9]\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]  => the original regex
   # 0000 1 2 333 4 555 6 7 0000  => value of "state" in each position
   # when processing string like "aaa 1.234.567.8 bbb"
   var state = 0
   var len = 0
   for i in 0 ..< text.len:
      let c = text[i]
      if c == '.':
         case state
         of {1, 3, 5}:
            state.inc
         of {2, 4, 6}:
            state = 0
      elif c >= '0' and c <= '9':
         case state
         of {0, 1}:
            state = 1
         of {2, 4}:
            state.inc
            len = 1
         of {3, 5}:
            if len == 3:
               state = 1
            else:
               len.inc
         of 6:
            return true
      else:
         state = 0
   result = false
