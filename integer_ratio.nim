# From: https://www.ics.uci.edu/~eppstein/numth/frap.c

## find rational approximation to given real number
## David Eppstein / UC Irvine / 8 Aug 1993
##
## With corrections from Arno Formella, May 2008
##
## usage: a.out r d
##   r is real number to approx
##   d is the maximum denominator allowed
##
## based on the theory of continued fractions
## if x = a1 + 1/(a2 + 1/(a3 + 1/(a4 + ...)))
## then best approximation is found by truncating this series
## (with some adjustments in the last term).
##
## Note the fraction can be recovered as the first column of the matrix
##  ( a1 1 ) ( a2 1 ) ( a3 1 ) ...
##  ( 1  0 ) ( 1  0 ) ( 1  0 )
## Instead of keeping the sequence of continued fraction terms,
## we just keep the last partial product of these matrices.

import os, strutils

proc main(params: seq[string]) =
   var m: array[2, array[2, int32]]
   var x, startx: float64
   var maxden: int32
   var ai: int32

   # read command line arguments
   if params.len != 2:
      quit("usage: ./script r d\n")  # argument missing

   startx = parseFloat(params[0])
   x = startx
   maxden = int32(parseInt(params[1]))

   # initialize matrix
   m[0][0] = 1
   m[1][1] = 1
   m[0][1] = 0
   m[1][0] = 0

   # loop finding terms until denom gets too big
   ai = int32(x)
   while m[1][0] * ai + m[1][1] <= maxden:
      swap m[0][1], m[0][0]
      swap m[1][0], m[1][1]
      m[0][0] = m[0][1] * ai + m[0][0]
      m[1][0] = m[1][1] * ai + m[1][0]
      if x == float64(ai): break  # division by zero
      x = 1/(x - float64(ai))
      if x > float64(0x7FFFFFFF): break  # representation failure
      ai = int32(x)

   # now remaining x is between 0 and 1/ai
   # approx as either 0 or 1/m where m is max that will fit in maxden
   # first try zero
   echo "$#/$#, error = $#\n".format(m[0][0], m[1][0],
      startx - (m[0][0] / m[1][0]))

   # now try other possibility
   ai = (maxden - m[1][1]) div m[1][0]
   m[0][0] = m[0][0] * ai + m[0][1]
   m[1][0] = m[1][0] * ai + m[1][1]
   echo "$#/$#, error = $#\n".format(m[0][0], m[1][0],
      startx - (m[0][0] / m[1][0]))

main(commandLineParams())
