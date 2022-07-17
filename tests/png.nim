# A simple class for parsing, serializing, and mutating an PNG file.
# https://en.wikipedia.org/wiki/Portable_Network_Graphics
# It is an example of a custom mutator for libFuzzer
# (https://llvm.org/docs/LibFuzzer.html) used for
# "structure-aware coverage-guided fuzzing".
import endians2

type
  PngMutator = object
    # Parse the input stream as a PNG file,
    # put every chunk into its own vector,
    # uncompress chunk data when needed,
    # merge the IDAT chunks into one vector.
    ihdr_: V
    chunks_: vector[PngMutatorChunk]

  V = seq[byte]

  PngMutatorChunk = object
    `type`: uint32
    v: V

proc newPngMutator(`in`: var Stream): PngMutator =
  result.ihdr_.grow(13, 0'b)
  Read4(`in`)
  Read4(`in`)
  setPosition(`in`, 8)
  # Skip the 8-byte magic value.
  # read IHDR.
  if ReadInteger(`in`) != 13:
    return
  if Read4(`in`) != Type("IHDR"):
    return
  `in`.read(cast[cstring](ihdr_.data()), ihdr_.size())
  Read4(`in`)
  # ignore CRC
  var idat_idx = -1
  while `in`:
    var len: uint32_t = ReadInteger(`in`)
    var `type`: uint32_t = Read4(`in`)
    if `type` == Type("IEND"):
      break
    var chunk_name: array[5, char]
    memcpy(chunk_name, addr(`type`), 4)
    chunk_name[4] = 0
    if len > (1 shl 20):
      return
    proc v(a1: len): V
    `in`.read(cast[cstring](v.data()), len)
    Read4(`in`)
    # ignore CRC
    if `type` == Type("IDAT"):
      if idat_idx != -1:
        Append(addr(chunks_[idat_idx].v), v)
      else:
        idat_idx = chunks_.size()
        chunks_.push_back((`type`, v))
    elif `type` == Type("iCCP"):
      var it: auto = v.begin()
      while it < v.`end`() and isprint(it[]):
        inc(it)
      if it < v.`end`() and not it[]:
        inc(it)
      if it < v.`end`() and not it[]:
        inc(it)
      v = V(it, v.`end`())
      var uncompressed: auto = Uncompress(v)
      chunks_.push_back((`type`, uncompressed))
      var compressed: auto = Compress(uncompressed)
    else:
      chunks_.push_back((`type`, v))
    # std::cerr << "CHUNK: " << chunk_name << std::endl;
  if idat_idx != -1:
    chunks_[idat_idx].v = Uncompress(chunks_[idat_idx].v)

proc Serialize*(this: var PngMutator; `out`: var ostream) =
  const header = [0x89'u8, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
  `out`.write(cast[cstring](header), sizeof(header))
  WriteChunk(`out`, "IHDR", ihdr_)
  for ch in chunks_:
    if ch.`type` == Type("iCCP"):
      var v: V
      v.push_back('x')
      # assuming the iCCP name doesn't matter.
      v.push_back(0)
      v.push_back(0)
      var compressed: auto = Compress(ch.v)
      Append(addr(v), compressed)
      WriteChunk(`out`, ch.`type`, v)
    else:
      WriteChunk(`out`, ch.`type`, ch.v)
  WriteChunk(`out`, "IEND", ())

type
  Mutator = proc (data: ptr UncheckedArray[byte], len, maxLen: int): int =

proc mutate(this: var PngMutator; m: Mutator; Seed: int64) =
  proc rnd(a1: Seed): minstd_rand
  var M: auto = (proc (v: ptr V): auto =
    if v.empty():
      v.resize(v.size() + 1 + rnd() mod 256)
    v.resize(m(v.data(), v.size(), v.size())))
  case rnd() mod 6               ##  Mutate IHDR.
  of 0:
    m(ihdr_.data(), ihdr_.size(), ihdr_.size())
  of 1:
    if not chunks_.empty():
      M(addr(chunks_[rnd() mod chunks_.size()].v))
  of 2:
    shuffle(chunks_.begin(), chunks_.`end`(), rnd)
  of 3:
    if not chunks_.empty():
      chunks_.erase(chunks_.begin() + rnd() mod chunks_.size())
  of 4:
    let types: UncheckedArray[cstring] = ["IATx", "sTER", "hIST", "sPLT", "mkBF", "mkBS",
                                      "mkTS", "prVW", "oFFs", "iDOT", "zTXt", "mkBT",
                                      "acTL", "iTXt", "sBIT", "tIME", "iCCP", "vpAg",
                                      "tRNS", "cHRM", "PLTE", "bKGD", "gAMA", "sRGB",
                                      "pHYs", "fdAT", "fcTL", "tEXt", "IDAT", "pCAL",
                                      "sCAL", "eXIf", "fUZz"]
    let n_types: csize_t = sizeof((types) div sizeof((types[0])))
    var `type`: uint32_t = if (rnd() mod 10 <= 8): Type(types[rnd() mod n_types]) else: cast[uint32_t](rnd())
    var len: csize_t = rnd() mod 256
    if `type` == Type("fUZz"):
      len = 16
    proc v(a1: len): V
    for b in v:
      b = rnd()
    var pos: csize_t = rnd() mod (chunks_.size() + 1)
    chunks_.insert(chunks_.begin() + pos, (`type`, v))
  of 5:
    var it: auto = find_if(chunks_.begin(), chunks_.`end`(), (proc (ch: Chunk): auto =
      return ch.`type` == Type("fUZz")))
    if it != chunks_.`end`():
      m(it.v.data(), it.v.size(), it.v.size())

proc CrossOver*(this: var PngMutator; p: PngMutator; Seed: cuint) =
  if p.chunks_.empty():
    return
  proc rnd(a1: Seed): minstd_rand
  var idx: csize_t = rnd() mod p.chunks_.size()
  var ch: var auto = p.chunks_[idx]
  var pos: csize_t = rnd() mod (chunks_.size() + 1)
  chunks_.insert(chunks_.begin() + pos, ch)

proc Append*(this: var PngMutator; to: ptr V; `from`: V) =
  to.insert(to.`end`(), `from`.begin(), `from`.`end`())

proc Read4*(this: var PngMutator; `in`: var istream): uint32_t =
  var res: uint32_t = 0
  `in`.read(cast[cstring](addr(res)), sizeof((res)))
  return res

proc ReadInteger*(this: var PngMutator; `in`: var Stream): uint32 =
  result = swapBytes(readUint32(`in`))

proc kind(tagname: string): uint32 =
  assert len(tagname) == 4
  copyMem(addr result, cstring(tagname), 4)

proc WriteInt*(this: var PngMutator; `out`: var ostream; x: uint32_t) =
  x = __builtin_bswap32(x)
  `out`.write(cast[cstring](addr(x)), sizeof((x)))

proc WriteChunk*(this: var PngMutator; `out`: var ostream; `type`: cstring; chunk: V;
                compress: bool = false) =
  var compressed: V
  let v: ptr V = addr(chunk)
  if compress:
    compressed = Compress(chunk)
    v = addr(compressed)
  var len: uint32_t = v.size()
  var crc: uint32_t = crc32(0, cast[ptr cuchar](`type`), 4)
  if v.size():
    crc = crc32(crc, cast[ptr cuchar](v.data()), v.size())
  WriteInt(`out`, len)
  `out`.write(`type`, 4)
  `out`.write(cast[cstring](v.data()), v.size())
  WriteInt(`out`, crc)

proc WriteChunk*(this: var PngMutator; `out`: var ostream; `type`: uint32_t; chunk: V) =
  var type_s: array[5, char]
  memcpy(type_s, addr(`type`), 4)
  type_s[4] = 0
  WriteChunk(`out`, type_s, chunk)

proc Uncompress*(this: var PngMutator; compressed: V): V =
  var v: V
  let kMaxBuffer: csize_t = 1 shl 28
  var sz: csize_t = compressed.size() * 4
  while sz < kMaxBuffer:
    v.resize(sz)
    var len: culong = sz
    var res: auto = uncompress(v.data(), addr(len), compressed.data(),
                           compressed.size())
    if res == Z_BUF_ERROR:
      sz = sz * 2
      continue
    if res != Z_OK:
      return ()
    v.resize(len)
    break
    sz = sz * 2
  return v

proc compress*(this: var PngMutator; uncompressed: V): V =
  var v: V
  let kMaxBuffer: csize_t = 1 shl 28
  var sz: csize_t = uncompressed.size()
  while sz < kMaxBuffer:
    v.resize(sz)
    var len: culong = sz
    var res: auto = compress(v.data(), addr(len), uncompressed.data(),
                         uncompressed.size())
    if res == Z_BUF_ERROR:
      sz = sz * 2
      continue
    if res != Z_OK:
      return ()
    v.resize(len)
    break
    sz = sz * 2
  return v

proc printHex*(this: var PngMutator; v: V; max_n: csize_t) =
  var i = 0
  while i < max_n and i < v.size():
    stderr.write "0x", hex, cast[cuint](v[i]), " ", dec
    inc(i)
  stderr.write endl

when defined(PNG_MUTATOR_DEFINE_LIBFUZZER_CUSTOM_MUTATOR):
  when isStandaloneTarget:
    proc mutate(data: ptr UncheckedArray[byte], len, maxLen: int): int =
      assert(false, "mutate should not be called from StandaloneFuzzTarget")
  else:
    proc mutate(data: ptr UncheckedArray[byte], len, maxLen: int): int {.
        importc: "LLVMFuzzerMutate".}

  proc customMutator(data: ptr UncheckedArray[byte]; len, maxLen: int, seed: int64): int {.
      exportc: "LLVMFuzzerCustomMutator".} =
    proc s(Data: reinterpret_cast[cstring]; a2: Size): string
    proc `in`(a1: s): stringstream
    var `out`: stringstream
    proc p(a1: `in`): PngMutator
    p.Mutate(LLVMFuzzerMutate, Seed)
    p.Serialize(`out`)
    let str: var auto = `out`.str()
    if str.size() > MaxSize:
      return Size
    memcpy(Data, str.data(), str.size())
    return str.size()

  proc customCrossOver(data1: openarray[byte], data2: openarray[byte],
      res: var openarray[byte], seed: int64): int {.
      exportc: "LLVMFuzzerCustomCrossOver".} =
    var
      in1 = newMemStream(data1)
      in2 = newMemStream(data2)
    var
      p1 = newPngMutator(in1)
      p2 = newPngMutator(in2)
    p1.CrossOver(p2, Seed)
    var `out`: stringstream
    p1.Serialize(`out`)
    let str: var auto = `out`.str()
    if str.len > res.len:
      return 0
    memcpy(Out, str.data(), str.size())
    return str.size()
