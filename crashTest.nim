import strutils

# proc bitLen(num: int): int =
#    # returns the number of bits necessary to represent an integer in binary
#    # excluding the sign and leading zeros.
#    var num = uint(num)
#    while num != 0:
#       result.inc
#       num = num shr 1
# 
# for i in low(int) .. low(int) + 20:
#    let len = bitLen(i)
#    echo len
#    let str = toBin(i, len)
#    echo str, " ", i
# 

for i in 0 .. 10:
   echo cast[bool](i)
#    echo not( not( i))
