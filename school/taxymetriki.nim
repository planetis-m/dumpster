import strutils, math

type
   HardChoice = enum
      None, Omoax, Epivatiko, Stadia, Dior 

proc tacheometry(horizontalDist: float; hardware: HardChoice) =
   # Υπολογισμός συντεταγμένων σημείων από ταχυμετρική αποτύπωση
   assert hardware != None
   # Εισαγωγή απόστασης σκοπευτικού άξονα θεοδόλιχου και άξονα οργάνου EDM
   let OrizAngle = # Δίνεται η γωνία διεύθυνσης προς το σκοπευόμενο σημείο
   if hardware == Epivatiko:
      let dyp = readLine("Απόσταση τηλεσκοπίου - EDM").parseFloat
      let ArcDz = sin(VertAngle) * Dyp
   var OrizDist = SlopeDist * sin(VertAngle)
   if hardware == Dior:
      OrizDist = SlopeDist * sin(VertAngle) - (0.087 * sin(Pi / 2 - VertAngle) / sin(VertAngle)
   if hardware == Stadia:
      OrizDist = OrizDist * sin(VertAngle)
   # Υπολογισμός συντεταγμένων σημείου
   let X = XStation + XKS * OrizDist * sin(OrizAngle)
   let Y = YStation + XKS * OrizDist * cos(OrizAngle)
   # Υπολογισμός υψόμετρου σημείου
   var H: float
   if TargHeight != -99:
      case hardware
      of Stadia:
         H = HStation + 0.5 * SlopeDist * sin(2 * VertAngle) + InstrHeight - TargHeight
         # TargHeight είναι η ένδειξη του μεσαίου νήματος επί της σταδίας
      of Dior:
         H = HStation + SlopeDist * cos(VertAngle) + InstrHeight + 0.087 * sin(VertAngle) - TargHeight
      else:
         H = HStation + SlopeDist * cos(VertAngle) + InstrHeight - TargHeight
   else:
      H = -999.0
