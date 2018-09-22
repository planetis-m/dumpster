
proc count_of_1bits(value: int): int =
   var value = value
   while value > 0:
      value = value and value - 1
      inc(result)

echo count_of_1bits(0b11001100)
