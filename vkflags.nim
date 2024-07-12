import macros, typetraits

type
  VkBufferUsageFlagBits* {.size: sizeof(int32).} = enum
    TransferSrcBit = 1
    TransferDstBit
    UniformTexelBufferBit = 4
    StorageTexelBufferBit = 8
    UniformBufferBit = 16
    StorageBufferBit = 32
    IndexBufferBit = 64
    VertexBufferBit = 128
    IndirectBufferBit = 256
    ConditionalRenderingBitExt = 512
    ShaderBindingTableBitKhr = 1024
    TransformFeedbackBufferBitExt = 2048
    TransformFeedbackCounterBufferBitExt = 4096
    VideoDecodeSrcBitKhr = 8192
    VideoDecodeDstBitKhr = 16384
    VideoEncodeDstBitKhr = 32768
    VideoEncodeSrcBitKhr = 65536
    ShaderDeviceAddressBit = 131072
    AccelerationStructureBuildInputReadOnlyBitKhr = 524288
    AccelerationStructureStorageBitKhr = 1048576
    SamplerDescriptorBufferBitExt = 2097152
    ResourceDescriptorBufferBitExt = 4194304
    MicromapBuildInputReadOnlyBitExt = 8388608
    MicromapStorageBitExt = 16777216
    ExecutionGraphScratchBitAmdx = 33554432
    PushDescriptorsDescriptorBufferBitExt = 67108864

  VkBufferUsageFlagBits2KHR* {.size: sizeof(int32).} = enum
    TransferSrcBit = 1
    TransferDstBit
    UniformTexelBufferBit = 4
    StorageTexelBufferBit = 8
    UniformBufferBit = 16
    StorageBufferBit = 32
    IndexBufferBit = 64
    VertexBufferBit = 128
    IndirectBufferBit = 256
    ConditionalRenderingBitExt = 512
    ShaderBindingTableBit = 1024
    TransformFeedbackBufferBitExt = 2048
    TransformFeedbackCounterBufferBitExt = 4096
    VideoDecodeSrcBit = 8192
    VideoDecodeDstBit = 16384
    VideoEncodeDstBit = 32768
    VideoEncodeSrcBit = 65536
    ShaderDeviceAddressBit = 131072
    AccelerationStructureBuildInputReadOnlyBit = 524288
    AccelerationStructureStorageBit = 1048576
    SamplerDescriptorBufferBitExt = 2097152
    ResourceDescriptorBufferBitExt = 4194304
    MicromapBuildInputReadOnlyBitExt = 8388608
    MicromapStorageBitExt = 16777216
    ExecutionGraphScratchBitAmdx = 33554432
    PushDescriptorsDescriptorBufferBitExt = 67108864

  VkFlags* = distinct uint32
  VkFlags64* = distinct uint64
  VkBufferUsageFlags* = distinct VkFlags
  VkBufferUsageFlags2KHR* = distinct VkFlags

macro flagsImpl(base, bits: typed, args: varargs[untyped]): untyped =
  let arr = newNimNode(nnkBracketExpr)
  for n in args: arr.add newCall(base, newDotExpr(bits, n))
  result = nestList(bindSym"or", arr)
  echo result.repr

template `{}`*(t: typedesc[VkBufferUsageFlags]; args: varargs[typed]): untyped =
  t(flagsImpl(uint32, VkBufferUsageFlagBits, args))

template `{}`*(t: typedesc[VkBufferUsageFlags2KHR]; args: varargs[typed]): untyped =
  t(flagsImpl(uint32, VkBufferUsageFlagBits2KHR, args))

# V1
let x = VkBufferUsageFlags{StorageBufferBit, UniformBufferBit}
let y = VkBufferUsageFlags2KHR{StorageBufferBit, IndexBufferBit}
