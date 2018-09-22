import math

let
   #starting position
   x1 = 0.0 #m
   y1 = 1.3 #m

   #final postion
   y2 = 0.0 #m

   #starting velocity
   v1 = 3.3 #m/s

   #vertical acceleration
   g = 9.8 #m/s^2

   #launch angle (in rad)
   theta = 35.0 * Pi / 180.0

   #calculate time (twice)
   c = y1 - y2
   a = -g / 2
   b = v1 * sin(theta)
   t1 = (-b + sqrt(b^2 - 4 * a * c))/(2 * a)
   t2 = (-b - sqrt(b^2 - 4 * a * c))/(2 * a)

echo("t = ", t1, " and ", t2, " s")

let
   x2a = x1 + v1 * cos(theta) * t1
   x2b = x1 + v1 * cos(theta) * t2

echo("x2 = ", x2a, " m")
echo("x2 = ", x2b, " m")
