
proc average(x, y: float): float =
   var samesign: bool
   if x >= 0.0:
      if y >= 0.0: 
         samesign = true
      else:
         samesign = false
   else:
      if y >= 0.0: 
         samesign = false
      else:
         samesign = true

   if samesign:
      if y >= x:
         x + ((y - x) / 2.0)
      else:
         y + ((x - y) / 2.0)
   else:
      (x + y) / 2.0

echo average(123123.0, 1221121.0)
