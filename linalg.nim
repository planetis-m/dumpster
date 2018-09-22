# Copyright 2016 UniCredit S.p.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright (C) 2017 <Antonis G.>

import math, random

# -----------------
# Type declarations
# -----------------

type
   OrderType* = enum
      rowMajor, colMajor
   Vector*[N: static[int]] = ref array[N, float]
   Matrix*[M, N: static[int]] = object
      order: OrderType
      data: ref array[M * N, float]
   Array[N: static[int]] = array[N, float]
   DoubleArray[M, N: static[int]] = array[M, array[N, float]]

# -------------------
# Initialize routines
# -------------------

proc matrix*[M, N](xs: DoubleArray[M, N], order = colMajor): Matrix[M, N] =
   result.order = order
   new result.data
   for i in 0 ..< M:
      for j in 0 ..< N:
         result.data[j * M + i] = xs[i][j]

proc eye*(N: static[int], order = colMajor): Matrix[N, N] =
   result.order = order
   new result.data
   for i in 0 ..< N:
      result.data[i * (1 + N)] = 1.0

proc randMatrix*[M, N](max = 1.0, order = colMajor): Matrix[M, N] =
   result.order = order
   new result.data
   for i in 0 ..< M * N:
      result.data[i] = rand(max)

# ---------------
# Access routines
# ---------------

proc at*[M, N](m: Matrix[M, N], i, j: int): float {.inline.} =
   let data = cast[ref DoubleArray[N, M]](m.data)
   data[j][i]

template `[]`*(m: Matrix, i, j: int): float = m.at(i, j)

proc row*[M, N](m: Matrix[M, N], i: int): Vector[N] {.inline.} =
   new result
   for j in 0 ..< N:
      result[j] = m.at(i, j)

# ----------------
# Display routines
# ----------------

proc toStringHorizontal[N](v: Vector[N]): string =
   result = "[ "
   for i in 0 .. N - 2:
      result.add $(v[i]) & "\t"
   result.add $(v[N - 1]) & " ]"

proc `$`*[M, N](m: Matrix[M, N]): string =
   result = "[ "
   for i in 0 .. M - 2:
      result.add toStringHorizontal(m.row(i)) & "\n  "
   result.add toStringHorizontal(m.row(M - 1)) & " ]"

# -------------------
# Operations routines
# -------------------

proc trace*[N](m: Matrix[N, N]): float =
   for i in 0 ..< N:
      result += m.data[i * (1 + N)]

proc `*`*[M, N](m: Matrix[M, N], k: float64): Matrix[M, N]  {.inline.} =
   result.order = m.order
   new result.data
   for i in 0 ..< M * N:
      result.data[i] = m.data[i] * k

template `*`*(k: float64, v: Vector or Matrix): untyped = v * k
template `/`*(v: Vector or Matrix, k: float64): untyped = v * (1 / k)

proc `*`*[M, N, K](a: Matrix[M, K], b: Matrix[K, N]): Matrix[M, N] {.inline.} =
   assert a.order == b.order
   result.order = a.order
   new result.data
   let
      a_data = cast[ref DoubleArray[K, M]](a.data)
      b_data = cast[ref DoubleArray[N, K]](b.data)
      res_data = cast[ref DoubleArray[N, M]](result.data)
   for i in 0 ..< M:
      for j in 0 ..< N:
         for k in 0 ..< K:
            res_data[j][i] += a_data[k][i] * b_data[j][k]

proc `+`*[M, N](a, b: Matrix[M, N]): Matrix[M, N] {.inline.} =
   assert a.order == b.order
   result.order = a.order
   new result.data
   let
      a_data = cast[ref DoubleArray[N, M]](a.data)
      b_data = cast[ref DoubleArray[N, M]](b.data)
      res_data = cast[ref DoubleArray[N, M]](result.data)
   for i in 0 ..< M:
      for j in 0 ..< N:
         res_data[j][i] = a_data[j][i] + b_data[j][i]

proc `-`*[M, N](a, b: Matrix[M, N]): Matrix[M, N] {.inline.} =
   assert a.order == b.order
   result.order = a.order
   new result.data
   let
      a_data = cast[ref DoubleArray[N, M]](a.data)
      b_data = cast[ref DoubleArray[N, M]](b.data)
      res_data = cast[ref DoubleArray[N, M]](result.data)
   for i in 0 ..< M:
      for j in 0 ..< N:
         res_data[j][i] = a_data[j][i] - b_data[j][i]

proc `=~`*[M, N: static[int]](a, b: Matrix[M, N]): bool {.inline.} =
   assert a.order == b.order
   const epsilon = 1e-8
   let
      a_data = cast[ref DoubleArray[N, M]](a.data)
      b_data = cast[ref DoubleArray[N, M]](b.data)
   for i in 0 ..< M:
      for j in 0 ..< N:
         if abs(a_data[j][i] - b_data[j][i]) > epsilon:
            return false
   return true

template `!=~`*(a, b: Matrix): bool =
   not (a =~ b)

proc det*(m: Matrix[3, 3]): float =
   + m[0,0] * (m[1,1] * m[2,2] - m[2,1] * m[1,2]) -
     m[1,0] * (m[0,1] * m[2,2] - m[2,1] * m[0,2]) +
     m[2,0] * (m[0,1] * m[1,2] - m[1,1] * m[0,2])

proc eigvals*(m: Matrix[3, 3]): tuple[eig1, eig2, eig3: float] =
   # https://en.wikipedia.org/wiki/Eigenvalue_algorithm
   let p1 = pow(m[0,1], 2.0) + pow(m[0,2], 2.0) + pow(m[1,2], 2.0)
   if p1 == 0:
      # m is diagonal.
      return (m[0,0], m[1,1], m[2,2])
   let
      q = trace(m) / 3.0
      p2 = pow(m[0,0]-q, 2.0) + pow(m[1,1]-q, 2.0) + pow(m[2,2]-q, 2.0) + 2.0 * p1
      p = sqrt(p2 / 6.0)
      b = (1.0 / p) * (m - q * eye(3))
      r = det(b) / 2.0
   # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
   # but computation error can leave it slightly outside this range.
   let phi =
      if r <= -1:
         Pi / 3.0
      elif r >= 1:
         0.0
      else:
         arccos(r) / 3.0
   # the eigenvalues satisfy eig3 <= eig2 <= eig1
   result.eig1 = q + 2.0 * p * cos(phi)
   result.eig3 = q + 2.0 * p * cos(phi + (2.0*Pi/3.0))
   result.eig2 = 3.0 * q - result.eig1 - result.eig3 # since trace(A) = eig1 + eig2 + eig3


when isMainModule:
   block scalarMultiply:
      let
         m1 = matrix([
            [1.0, 3.0],
            [2.0, 8.0],
            [-2.0, 3.0]
         ])
         m2 = matrix([
            [3.0, 9.0],
            [6.0, 24.0],
            [-6.0, 9.0]
         ])
      let ans1 = m1 * 3.0
      let ans2 = 3.0 * m1
      assert(ans1 =~ m2)
      assert(ans2 =~ m2)

   block scalarDivide:
      let
         m1 = matrix([
            [1.0, 3.0],
            [2.0, 8.0],
            [-2.0, 3.0]
         ])
         m2 = matrix([
            [3.0, 9.0],
            [6.0, 24.0],
            [-6.0, 9.0]
         ])
      let ans = m2 / 3.0
      assert(ans =~ m1)

   block multiply:
      let
         m1 = matrix([
            [1.0, 1.0, 2.0, -3.0],
            [3.0, 0.0, -7.0, 2.0]
         ])
         m2 = matrix([
            [1.0, 1.0, 2.0],
            [3.0, 1.0, -5.0],
            [-1.0, -1.0, 2.0],
            [4.0, 2.0, 3.0]
         ])
         m3 = matrix([
            [-10.0, -6.0, -8.0],
            [18.0, 14.0, -2.0]
         ])
      let ans = m1 * m2
      assert(ans =~ m3)

   block add:
      let
         m1 = matrix([
            [1.0, -1.0],
            [-2.0, 3.0]
         ])
         m2 = matrix([
            [1.0, 2.0],
            [3.0, 4.0]
         ])
         m3 = matrix([
            [2.0, 1.0],
            [1.0, 7.0]
         ])
      let ans = m1 + m2
      assert(ans =~ m3)

   block subtract:
      let
         m1 = matrix([
            [1.0, -1.0],
            [-2.0, 3.0]
         ])
         m2 = matrix([
            [1.0, 2.0],
            [3.0, 4.0]
         ])
         m3 = matrix([
            [0.0, -3.0],
            [-5.0, -1.0]
         ])
      let ans = m1 - m2
      assert(ans =~ m3)

   block randomMatrix:
      let m1 = randMatrix[3, 5]()
      for i in 0 ..< m1.M:
         for j in 0 ..< m1.N:
            assert m1[i, j] < 1.0
