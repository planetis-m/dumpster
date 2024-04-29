import vulkan, std/envvars

const float16Int8Extension = "VK_KHR_shader_float16_int8"

template toCString(arr: array): cstring = cast[cstring](addr arr)
template toCStringArray(arr: array): cstringArray = cast[cstringArray](addr arr)

proc main =
  vkPreload()

  # Set the AMDVLK_ENABLE_DEVELOPING_EXT environment variable
  putEnv("AMDVLK_ENABLE_DEVELOPING_EXT", float16Int8Extension)

  # Get the extension count
  var extensionCount: uint32
  discard vkEnumerateInstanceExtensionProperties(nil, extensionCount.addr, nil)
  echo "Number of available extensions: ", extensionCount

  # Get the list of available extensions
  var extensions = newSeq[VkExtensionProperties](extensionCount)
  discard vkEnumerateInstanceExtensionProperties(nil, extensionCount.addr, extensions[0].addr)

  # Check if the extension is supported
  var isFloat16Int8Supported = false
  for ext in extensions:
    if ext.extensionName.toCString == float16Int8Extension.cstring:
      isFloat16Int8Supported = true
      break

  if isFloat16Int8Supported:
    echo "VK_KHR_shader_float16_int8 extension is supported!"
  else:
    echo "VK_KHR_shader_float16_int8 extension is not supported."

  # Create a Vulkan instance
  let enabledExtensions = [float16Int8Extension.cstring]
  let instanceCreateInfo = newVkInstanceCreateInfo(
    flags = 0.VkInstanceCreateFlags,
    pApplicationInfo = nil,
    enabledLayerCount = 0,
    ppEnabledLayerNames = nil,
    enabledExtensionCount = 1,
    ppEnabledExtensionNames = enabledExtensions.toCStringArray
  )

  var instance: VkInstance
  if vkCreateInstance(instanceCreateInfo.addr, nil, instance.addr) != VkSuccess:
    quit("Failed to create Vulkan instance")
  else:
    echo "Successfully created Vulkan instance!"

  discard vkInit(instance)

  # Clean up
  vkDestroyInstance(instance, nil)

main()
