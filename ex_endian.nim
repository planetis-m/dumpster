# https://betterexplained.com/articles/understanding-big-and-little-endian-byte-order/
import strutils

var a = [0x12'u8, 0x34, 0x56, 0x78]
var p = addr a
echo "array[4, int8]:"
var buf = ""
for i in 0..<4:
  buf.add toHex(cast[ptr UncheckedArray[int8]](p)[i], 2)
buf = insertSep(buf, ' ', 2)
echo spaces(2), buf

buf = ""
echo "array[2, int16]:"
for i in 0..<2:
  buf.add toHex(cast[ptr UncheckedArray[int16]](p)[i], 4)
buf = insertSep(buf, ' ', 2)
echo spaces(2), buf

buf = ""
echo "int32:"
buf.add toHex(cast[ptr int32](p)[], 8)
buf = insertSep(buf, ' ', 2)
echo spaces(2), buf
