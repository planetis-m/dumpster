type
  float16* = distinct uint16

proc fromHalf*(x: float16): float32 =
  ## Convert half-float (stored as unsigned short) to float
  ## Ref: https://stackoverflow.com/a/60047308
  let e = (x.uint32 and 0x7c00) shr 10'u32 # Exponent
  let m = (x.uint32 and 0x03ff) shl 13'u32 # Mantissa
  let fm = float32(m)
  # Evil log2 bit hack to count leading zeros in denormalized format
  let v = cast[uint32](fm) shr 23'u32
  # sign : normalized : denormalized
  let r = (x.uint32 and 0x8000) shl 16'u32 or uint32(e != 0)*((e + 112) shl 23'u32 or m) or
      uint32((e == 0) and (m != 0))*((v - 37) shl 23'u32 or ((m shl (150 - v)) and 0x007fe000))
  result = cast[float32](r)

proc toHalf*(x: float32): float16 =
  ## Convert float to half-float (stored as unsigned short)
  # Round-to-nearest-even: add last bit after truncated mantissa
  let b = cast[uint32](x) + 0x00001000
  let e = (b and 0x7f800000) shr 23'u32 # Exponent
  # Mantissa in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
  let m = b and 0x007fffff
  # sign : normalized : denormalized : saturate
  let r = uint16((b and 0x80000000'u32) shr 16'u32 or
      uint32(e > 112)*((((e - 112) shl 10'u32) and 0x7c00) or (m shr 13'u32)) or
      (uint32(e < 113) and uint32(e > 101))*((((0x007ff000 + m) shr (125 - e)) + 1) shr 1'u32) or
      uint32(e > 143)*0x7fff)
  result = float16(r)

when isMainModule:
  let x = toHalf(0.125'f32)
  echo fromHalf(x)
