proc date(d, m, y: int) =
   var c = 0
   check1(d, m, y, c)
   if c == 0:
      echo "Αρνητικές τιμές"
   check2(d, m, y, c)
   if c == 0:
      echo "Ανύπαρκτη ημερομηνία"
   check3(d, m, y, c)
   if c == 0:
      echo "Λανθασμένη ημερομηνία"

proc check1(d, m, y: int; c: var int) =
   # Αρνητικές ημερομηνίες
   c = if y > 0 and m > 0 and d > 0: 1 else: 0

proc check2(d, m, y: int; c: var int) =
   # Ιουλιανό ημερολόγιο
   c = if (10 < d and d < 23) and m == 3 and y = 1923: 0 else: 1

proc check3(d, m, y: int; c: var int) =
# Έλεγχος δίσεκτου έτους (αν δ=1 ΔΙΣΕΚΤΟ)
e4 <- e mod 4
e100 <- e mod 100
e400 <- e mod 400
d = 0
ΑΝ ε4 == 0 ΤΟΤΕ
   ΑΝ ε100=0 ΤΟΤΕ
      ΑΝ ε400=0 ΤΟΤΕ
      δ <- 1
      ΤΕΛΟΣ_ΑΝ
   ΑΛΛΙΩΣ
         δ <- 1
   ΤΕΛΟΣ_ΑΝ
ΤΕΛΟΣ_ΑΝ
# Ελεγχος αποδεκτής ημερομηνίας (c=1 ΑΠΟΔΕΚΤΗ)
   c = 0
   case m
   of 1, 3, 5, 7, 8, 10, 12:
      if d <= 31: c = 1
   of 4, 6, 9, 11:
      if d <= 30: c = 1
   of 2:
      if == 0 and d <= 28: c = 1
      if δ == 1 and d <= 29: c = 1
   else:
      discard
