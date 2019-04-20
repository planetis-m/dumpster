proc power(a, b: float): float =
   var power: array[100, float]
   # αποθήκευση στοιχείων πίνακα
   power[1] = a
   var i = 1
   var pow = 1
   while pow < b:
      i.inc
      pow = 2 * pow
      power[i] = power[i-1] * power[i-1]
   # ανεύρεση της δύναμης
   var used = 0
   result = 1
   while used < b:
      if used + pow <= b:
         result = result * power[i]
         used = used + pow
      pow = pow / 2
      i.dec
