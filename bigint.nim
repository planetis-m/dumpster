# http://faculty.cse.tamu.edu/djimenez/ut/utsa/cs3343/lecture20.html


const
   MaxDigits = 16
   Power = 9
   Base = 1000_000_000

type
   BigInt = object
      d: array[MaxDigits, int]

proc toBigInt(n: int): BigInt =
   # put the normal int n into the big int A
   var n = n
   # start indexing at the 0's place
   var i = 0
   # while there is still something left to the number
   # we're encoding...
   while n > 0:
      # put the least significant digit of n into A[i]
      result.d[i] = n mod Base
      i.inc
      # get rid of the least significant digit,
      # i.e., shift right once
      n = n div Base
   # fill the rest of the array up with zeros
   while i < MaxDigits:
      result.d[i] = 0
      i.inc

proc toBigInt(s: string): BigInt =
   # Μετατρέπει το str σε bignum. Το str είναι βάση 10,
   # και το most significant bit είναι το δεξιότερο
   let slen = s.len
   assert slen <= Power * MaxDigits

   var slow = 0
   var powr: array[Power, int]
   powr[0] = 1
   if s[0] == '-':
      powr[0] = -1
      slow = 1

   for i in 1 ..< Power:
      powr[i] = powr[i - 1] * 10

   for i in slow ..< slen:
      result.d[i div Power] += powr[i mod Power] * (int(s[
            slen - i - 1]) - int('0'))

proc inc(a: var BigInt) =
   # a.inc
   var i = 0
   # start indexing at the least significant digit
   while i < MaxDigits:
      # increment the digit
      a.d[i].inc
      # if it overflows (i.e., it was 9, now it's 10, too
      # big to be a digit) then...
      if a.d[i] == Base:
         # make it zero and index to the next 
         # significant digit
         a.d[i] = 0
         i.inc
      else:
         # otherwise, we are done
         break

proc `+`(a, b: BigInt): BigInt =
   # c = a + b
   # no carry yet
   var carry = 0
   # go from least to most significant digit
   var sum = 0
   for i in 0 ..< MaxDigits:
      # the i'th digit of C is the sum of the
      # i'th digits of A and B, plus any carry
      sum = a.d[i] + b.d[i] + carry
      # if the sum exceeds the base, then we have a carry.
      if sum >= Base:
         carry = 1
         # make sum fit in a digit (same as sum %= BASE)
         sum -= Base
      else:
         # otherwise no carry
         carry = 0
      # put the result in the sum
      result.d[i] = sum
   # if we get to the end and still have a carry, we don't have
   # anywhere to put it, so panic!
   if carry > 0:
      raise newException(OverflowError, "overflow in addition!")

proc `*`(a: BigInt; n: int): BigInt =
   # b = n * a
   # no extra overflow to add yet
   var carry = 0
   # for each digit, starting with least significant...
   var prod = 0
   for i in 0 ..< MaxDigits:
      # multiply the digit by n, putting the result in B
      prod = n * a.d[i]
      # add in any overflow from the last digit
      prod += carry
      # if this product is too big to fit in a digit...
      if prod >= Base:
         # handle the overflow
         carry = prod div Base
         prod = prod mod Base
      else:
         # no overflow
         carry = 0
      # put the result in the product
      result.d[i] = prod
   if carry > 0:
      raise newException(OverflowError, "overflow in multiplication!")

proc shiftLeft(a: var BigInt; n: int) =
   # "multiplies" a number by Base n
   # going from left to right, move everything over to the
   # left n spaces
   var i = MaxDigits - 1
   while i >= n:
      a.d[i] = a.d[i - n]
      i.dec
   # fill the last n digits with zeros
   while i >= 0:
      a.d[i] = 0
      i.dec

proc `*`(a, b: BigInt): BigInt =
   # c = a * b
   # c will accumulate the sum of partial products.
   # for each digit in A...
   for i in 0 ..< MaxDigits:
      # multiply B by digit A[i]
      var p = b * a.d[i]
      # shift the partial product left i bytes
      shiftLeft(p, i)
      # add result to the running sum
      result = result + p

proc `$`(a: BigInt): string =
   # Τυπώνει τον αριθμό
   template fill(num): untyped =
      var res = num
      for i in res.len ..< Power:
         res.add '0'
      result.add res

   var beginning = true
   for i in countdown(MaxDigits - 1, 0):
      if beginning and a.d[i] == 0:
         continue
      if beginning:
         beginning = false
         result.add $a.d[i]
      else:
         if a.d[i] < 0:
            fill $(-a.d[i])
         else:
            fill $a.d[i]
   if beginning:
      result.add '0'


var c = toBigInt(
      "23141234108799696796034572933423294546701973123586766728027843513098401103841834138130980542778523561234532672591958092174950757207405235423536")
echo c

var d = toBigInt("1000000000")
d = d * 2
echo d

var i = 0
while i < 10:
   c = c * d
   i.inc

echo c
