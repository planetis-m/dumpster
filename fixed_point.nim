##[
Fixed point arithmetic

A fixed point number is an alternative representation for a real number.
IEEE floats, `f32` and `f64`, being the standard format in processors with
Floating Point Units (FPU). You should consider using fixed numbers on
systems where there's no FPU and performance is critical as fixed point
arithmetic is faster than software emulated IEEE float arithmetic. Do note
that fixed point numbers tend to be more prone to overflows as they operate
in ranges much smaller than floats.

The fixed point numbers exposed in this library use the following naming
convention: `IxFy`, where `x` is the number of bits used for the integer
part and `y` is the number of bits used for the fractional part.

Unlike IEEE floats, fixed points numbers have *fixed* precision. One can
exchange range for precision by selecting different values for `x` and `y`:

- Range: `[-2 ^ (x - 1), 2 ^ (x - 1) - 2 ^ (-y)]`
- Precision: `2 ^ (-y)`

For example, the type `I1F7` has range `[-1, 0.9921875]` and precision
`0.0078125`.

- Casts

.. code-block:: Nim
   import fpa
   # 32-bit fixed point number, 16 bits for the integer part and 16 bits for
   # the fractional part
   type I16F16 = FixedPoint[int32, 16]
   template toI16F16(x: float): I16F16 = fromFloat[int32, 16](x)
   template toI16F16(x: int): I16F16 = fromInt[int32, 16](x)

   # casts an integer into a fixed point number (infallible)
   let q = toI16F16(1)
   # casts the fixed point number into a float (infallible)
   let f = toFloat(q)
   assert f == 1.0

- Arithmetic

.. code-block:: Nim
   # NOTE the `float` -> `I16F16` cast is fallible because of NaN and infinity
   assert toI16F16(1.25) + toI16F16(2.75) == toI16F16(4.0)
   assert toI16F16(2.0) / toI16F16(0.5) == toI16F16(4.0)
   assert toI16F16(2.0) * toI16F16(0.5) == toI16F16(1.0)

- Trigonometry

.. code-block:: Nim
   type I8F24 = FixedPoint[int32, 24]
   template toI8F24(x: float): I8F24 = fromFloat[int32, 24](x)

   let (r, _) = toI8F24(0.3).polar(toI8F24(0.4))
   assert abs(toFloat(r) - 0.5) < 1e-5
]##

type
   FixedPoint*[R: SomeSignedInt; F: static[int]] = object
      ## Fixed point number
      ##
      ## - `R` is the integer primitive used to stored the number
      ## - `F` is the number of bits used for the fractional part of the number
      bits: R

proc cmp*[R, F](x, y: FixedPoint[R, F]): int =
   result = cmp(x.bits, y.bits)

proc `==`*[R, F](x, y: FixedPoint[R, F]): bool =
   x.bits == y.bits

proc `<=`*[R, F](x, y: FixedPoint[R, F]): bool =
   x.bits <= y.bits

proc `<`*[R, F](x, y: FixedPoint[R, F]): bool =
   x.bits < y.bits

proc `-`*[R, F](x: FixedPoint[R, F]): FixedPoint[R, F] =
   result = FixedPoint[R, F](bits: -x.bits)

proc `+`*[R, F](x, y: FixedPoint[R, F]): FixedPoint[R, F] =
   result = FixedPoint[R, F](bits: x.bits + y.bits)

proc `-`*[R, F](x, y: FixedPoint[R, F]): FixedPoint[R, F] =
   result = FixedPoint[R, F](bits: x.bits - y.bits)

proc `+=`*[R, F](x: var FixedPoint[R, F], y: FixedPoint[R, F]) =
   x.bits += y.bits

proc `-=`*[R, F](x: var FixedPoint[R, F], y: FixedPoint[R, F]) =
   x.bits -= y.bits

template biggerInt(x: untyped): untyped =
   when x is int8: int16(x)
   elif x is int16: int32(x)
   elif x is int32: int64(x)
   else: x #.toInt128

proc `*`*[R, F](x, y: FixedPoint[R, F]): FixedPoint[R, F] =
   when defined(useBiggerInt):
      result = FixedPoint[R, F](bits: R(biggerInt(x.bits) * biggerInt(y.bits) shr F))
   else:
      const halfF = F div 2
      result = FixedPoint[R, F](bits: (x.bits shr halfF) * (y.bits shr halfF))

proc `/`*[R, F](x, y: FixedPoint[R, F]): FixedPoint[R, F] =
   assert y.bits != 0, "Division by zero"
   when defined(useBiggerInt):
      result = FixedPoint[R, F](bits: R(biggerInt(x.bits) shl F div y.bits))
   else:
      const halfF = F div 2
      result = FixedPoint[R, F](bits: (x.bits shl halfF) div y.bits shl halfF)

proc `*=`*[R, F](x: var FixedPoint[R, F], y: FixedPoint[R, F]) =
   x = x * y

proc `/=`*[R, F](x: var FixedPoint[R, F], y: FixedPoint[R, F]) =
   x = x / y

proc fromFloat*[R, F](x: float): FixedPoint[R, F] =
   result = FixedPoint[R, F](bits: R(x * float(1 shl F)))

proc fromFloat32*[R, F](x: float32): FixedPoint[R, F] =
   result = FixedPoint[R, F](bits: R(x * float32(1 shl F)))

proc fromInt*[R, F](x: int): FixedPoint[R, F] =
   result = FixedPoint[R, F](bits: R(x shl F))

proc fromInt32*[R, F](x: int32): FixedPoint[R, F] =
   result = FixedPoint[R, F](bits: R(x shl F))

proc toFloat*[R, F](x: FixedPoint[R, F]): float =
   result = float(x.bits) / float(1 shl F)

proc toFloat32*[R, F](x: FixedPoint[R, F]): float32 =
   result = float32(x.bits) / float32(1 shl F)

proc toInt*[R, F](x: FixedPoint[R, F]): int =
   result = int(x.bits shr F)

proc toInt32*[R, F](x: FixedPoint[R, F]): int32 =
   result = int32(x.bits shr F)

when isMainModule:
   type I16F16 = FixedPoint[int32, 16]

   template toI16F16(x: float32): I16F16 = fromFloat32[int32, 16](x)
   template toI16F16(x: int32): I16F16 = fromInt32[int32, 16](x)

   let f1 = toI16F16(126'i32)
   let f2 = toI16F16(3'i32)

   assert toFloat32(f1 / f2) == 42.0'f32
   assert toInt32(f1 * f2) == 378'i32
   assert toI16F16(1.25'f32) + toI16F16(2.75'f32) == toI16F16(4.0'f32)
