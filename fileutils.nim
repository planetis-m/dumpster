import os

proc readData*(f: File; buffer: pointer; bufLen: int): int =
   result = readBuffer(f, buffer, bufLen)

proc peekData*(f: File; buffer: pointer; bufLen: int): int =
   let pos = int(getFilePos(f))
   defer: setFilePos(f, pos)
   result = readBuffer(f, buffer, bufLen)

proc writeData(f: File; buffer: pointer; bufLen: int) =
   if writeBuffer(f, buffer, bufLen) != bufLen:
      raise newException(IOError, "cannot write to stream")

proc read[T](f: File; result: var T) =
   if readData(f, addr(result), sizeof(T)) != sizeof(T):
      raise newException(IOError, "cannot read from stream")

proc peek[T](f: File; result: var T) =
   if peekData(f, addr(result), sizeof(T)) != sizeof(T):
      raise newException(IOError, "cannot read from stream")

proc write*[T](f: File; x: T) =
   var y: T
   shallowCopy(y, x)
   writeData(f, addr(y), sizeof(y))

proc writeSeq*[T](f: File; x: seq[T]) =
   if x.len > 0: writeData(f, unsafeAddr x[0], x.len*sizeof(T))

proc readChar*(f: File): char =
   if readData(f, addr(result), sizeof(result)) != 1: result = '\0'

proc peekChar*(f: File): char =
   if peekData(f, addr(result), sizeof(result)) != 1: result = '\0'

proc readBool*(f: File): bool =
   read(f, result)

proc peekBool*(f: File): bool =
   peek(f, result)

proc readInt8*(f: File): int8 =
   read(f, result)

proc peekInt8*(f: File): int8 =
   peek(f, result)

proc readInt16*(f: File): int16 =
   read(f, result)

proc peekInt16*(f: File): int16 =
   peek(f, result)

proc readInt32*(f: File): int32 =
   read(f, result)

proc peekInt32*(f: File): int32 =
   peek(f, result)

proc readInt64*(f: File): int64 =
   read(f, result)

proc peekInt64*(f: File): int64 =
   peek(f, result)

proc readUInt8*(f: File): uint8 =
   read(f, result)

proc peekUInt8*(f: File): uint8 =
   peek(f, result)

proc readUInt16*(f: File): uint16 =
   read(f, result)

proc peekUInt16*(f: File): uint16 =
   peek(f, result)

proc readUInt32*(f: File): uint32 =
   read(f, result)

proc peekUInt32*(f: File): uint32 =
   peek(f, result)

proc readUInt64*(f: File): uint64 =
   read(f, result)

proc peekUInt64*(f: File): uint64 =
   peek(f, result)

proc readFloat32*(f: File): float32 =
   read(f, result)

proc peekFloat32*(f: File): float32 =
   peek(f, result)

proc readFloat64*(f: File): float64 =
   read(f, result)

proc peekFloat64*(f: File): float64 =
   peek(f, result)

proc readBytes*(f: File; len: int): seq[byte] =
   result = newSeq[byte](len)
   let L = readData(f, addr(result[0]), len)
   if L != len: setLen(result, L)

proc peekBytes*(f: File; len: int): seq[byte] =
   result = newSeq[byte](len)
   let L = peekData(f, addr(result[0]), len)
   if L != len: setLen(result, L)

proc readAllBytes*(f: File): seq[byte] =
   const bufferSize = 1000
   result = newSeq[byte](bufferSize)
   var r = 0
   while true:
      let readBytes = readData(f, addr(result[r]), bufferSize)
      if readBytes < bufferSize:
         setLen(result, r + readBytes)
         break
      inc(r, bufferSize)
      setLen(result, r + bufferSize)
