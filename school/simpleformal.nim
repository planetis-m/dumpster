# Δίδεται ταξινομημένος μονοδιάστατος πίνακας ακεραίων αριθμών
# 1000 θέσεων. Ζητείται να κατασκευαστεί πρόγραμμα, το οποίο να
# βρίσκει τη συχνότητα εμφάνισης κάθε αριθμού του πίνακα. Τα αποτε-
# λέσματα να εμφανίζονται στην οθόνη.

const a = [2, 2, 3, 3, 3, 3, 5, 6, 6, 6, 6, 6,
           7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9]

proc frequency1(a: array[25, int]) =
   var
      i = 0
      sum = 0
      previous = a[i]
   while i < a.len:
      if previous != a[i]:
         echo previous, sum
         previous = a[i]
         sum = 0
      sum.inc
      i.inc
   echo previous, sum

proc frequency2(a: array[25, int]) =
   var
      i = 0
      sum = 0
      previous = a[i]
   while i < a.len:
      while i < a.len and previous == a[i]:
         sum.inc
         i.inc
      echo previous, sum
      previous = a[i]
      sum = 0

proc frequency3(a: array[25, int]) =
   var
      i = 0
      sum = 0
      previous = a[i]
   while i < a.len:
      if previous == a[i]:
         sum.inc
         i.inc
      else:
         echo previous, sum
         previous = a[i]
         sum = 0
   echo previous, sum

frequency2(a)
