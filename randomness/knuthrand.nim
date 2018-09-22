# for random Knuth - from Knuth TAOCP Vol. 2 Seminumerical Algorithms section 3.6
# these numbers are really important - please do not change them, NEVER!!!
# if you want, write a new random number generator routine, with other name
# I think I found a minor possible improvement, the author said: if(z<=0) then z+=mm,
# but I think this would be better: if(z<=0) then z+=mm-1. - Yes, the author confirmed
import times

const
   mm = 2147483647
   aa = 48271
   qq = 44488
   rr = 3399
   mmm = 2147483399
   aaa = 40692
   qqq = 52774
   rrr = 3791

var
   xx: int
   yy: int
   zz: int

proc randomize*() =
   var tt = int(times.epochTime() * 1000_000_000)
   # xx is the current time
   # xx = 1 + ( (unsigned(tt)) % (unsigned(mm-1)) );
   xx = 1 + tt mod (mm - 1)
   # yy is the next random, after initializing yy with the current time
   # yy = 1 + ( (unsigned(tt)) % (unsigned(mmm-1)) );
   yy = 1 + tt mod (mmm - 1)
   yy = aaa * (yy mod qqq) - rrr * (yy div qqq)
   if yy < 0:
      inc(yy, mmm)
   zz = xx - yy
   if zz <= 0:
      inc(zz, mm - 1)

proc rand1mm1*(): int =
   xx = aa * (xx mod qq) - rr * (xx div qq)
   if xx < 0:
      inc(xx, mm)
   yy = aaa * (yy mod qqq) - rrr * (yy div qqq)
   if yy < 0:
      inc(yy, mmm)
   zz = xx - yy
   if zz <= 0:
      inc(zz, mm - 1)
   return zz

proc rand*(k: int): int =
   # like in Knuth TAOCP vol.2, reject some numbers (very few),
   # so that the distribution is perfectly uniform
   while true:
      let u = rand1mm1()
      if u <= k * ((mm - 1) div k):
         return u mod k

when isMainModule:
   block testCase:
      let cases =
         [(123, 123, 5937333, 5005116, 932217),
          (4321, 54321, 208578991, 62946733, 145632258),
          (87654321, 987654321, 618944401, 1625301246, 1141126801),
          (1, 1, 48271, 40692, 7579),
          (mm-1, mmm-1, 2147435376, 2147442707, 2147476315),
          (100, 1000, 4827100, 40692000, 2111618746)]

      for ixx, iyy, txx, tyy, tuu in items(cases):
         xx = ixx
         yy = iyy
         assert(rand1mm1() == tuu)
         assert(xx == txx)
         assert(yy == tyy)
