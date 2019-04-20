
func sameSign(x, y: float): bool =
   x >= 0.0 == y >= 0.0

func average(x, y: float): float =
   if sameSign(x, y):
      if y >= x:
         x + ((y - x) / 2.0)
      else:
         y + ((x - y) / 2.0)
   else:
      (x + y) / 2.0

echo average(123123.0, 1221121.0)
