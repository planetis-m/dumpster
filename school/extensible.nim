# Δίνεται ζεύγος αριθμών από το πληκτρολόγιο και ζητείται να εμφανί-
# ζονται με αύξουσα σειρά στην οθόνη.

proc sequence2a(a, b: float) =
   if a < b:
      echo a, b
   else:
      echo b, a

# Δίνεται τριάδα αριθμών από το πληκτρολόγιο και ζητείται να εμφανι-
# στούν με αύξουσα σειρά στην οθόνη.

proc sequence3a(a, b, c: float) =
   if a < b:
      if b < c:
         echo a, b, c
      else:
         if a < c:
            echo a, c, b
         else:
            echo c, a, b
   else:
      if a < c:
         echo b, a, c
      else:
         if b < c:
            echo b, c, a
         else:
            echo c, b, a

proc sequence2b(a, b: float) =
   var
      a = a
      b = b
   if a > b:
      swap a, b
   echo a, b

proc sequence3b(a, b, c: float) =
   var
      a = a
      b = b
      c = c
      s = true
   while s:
      s = false
      if a > b:
         swap a, b
         s = true
      if b > c:
         swap b, c
         s = true
   echo a, b
