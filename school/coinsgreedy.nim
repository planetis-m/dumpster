const c: array[15, int] = [
   1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000]

proc coins(c: array[15, int]; n, amount: int): int =
   var find = amount
   var choice = n
   while choice > 0 and find > 0:
      if c[choice] <= find:
         result.inc
         find = find - c[choice]
      else:
         choice.dec
