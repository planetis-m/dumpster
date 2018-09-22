import strutils, math

const
   plusSign = "+"
   minusSign = "-"
   timesSign = "*"
   intoSign = "/"
   powerSign = "^"


proc postfix(expression: string): float =
   var stack = @[]

   for x in expression.split():
      if x in ops:
         x = ops[x](stack.pop(-2), stack.pop(-1))
      else:
         x = parseFloat(x)
      stack.add(x)

   return stack.pop()


echo postfix("1 2 + 4 3 - + 10 5 / *")
