# https://www.youtube.com/watch?v=tAbBID3N64A
type
   ComparisonType = enum
      Equals, GreaterThan, LessThanEqual

   CriterionStatic = object
      fa: float
      ctype: ComparisonType

func compare1(this: CriterionStatic; x: float): bool =
   case this.ctype
   of Equals:
      x == fa
   of GreaterThan:
      x > fa
   of LessThanEqual:
      x <= fa

type
   CriterionStatic = object
      fa, fb: float

func compare2(this: CriterionStatic; x: float): bool =
   x >= fa and fb >= x

