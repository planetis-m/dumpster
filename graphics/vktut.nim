import sdl2, cstr, ../repos/vulkanim/vulkan
{.link: "/usr/lib/libvulkan.so".}

proc main =
   let
      windowWidth = 1366'i32
      windowHeight = 768'i32

   doAssert sdl2.init(INIT_VIDEO or INIT_EVENTS) == SdlSuccess

   let window = createWindow("SDL/Vulkan Skeleton", SDL_WINDOWPOS_UNDEFINED,
         SDL_WINDOWPOS_UNDEFINED, windowWidth, windowHeight, SDL_WINDOW_VULKAN)

   var extensionCount = 0'u32
   doAssert vulkanGetInstanceExtensions(window, addr extensionCount, nil) == True32

   var extensionNames = newCStringArray(extensionCount)
   doAssert vulkanGetInstanceExtensions(window, addr extensionCount, extensionNames.impl) == True32

   var appInfo = VkApplicationInfo(
      sType: vkStructureTypeApplicationInfo,
      pApplicationName: "SDL/Vulkan Skeleton",
      applicationVersion: vkMakeVersion(0, 0, 1),
      pEngineName: "No Engine",
      engineVersion: vkMakeVersion(0, 0, 1),
      apiVersion: vkApiVersion10)

   var createInfo = VkInstanceCreateInfo(
      sType: vkStructureTypeInstanceCreateInfo,
      pApplicationInfo: addr appInfo,
      enabledExtensionCount: extensionCount,
      ppEnabledExtensionNames: extensionNames.impl)

   var instance = vkNullHandle.VkInstance
   doAssert vkCreateInstance(addr createInfo, nil, addr instance) == vkSuccess

   var surface = vkNullHandle.VkSurfaceKHR
   doAssert vulkanCreateSurface(window, instance, addr surface) == True32

   var physicalDevice = vkNullHandle.VkPhysicalDevice
   var queueFamilyIndex: uint32
   var queue: VkQueue
   var device = vkNullHandle.VkDevice

   var queuePriority = 1.0'f32
   var deviceQueueCreateInfo = VkDeviceQueueCreateInfo(
      sType: vkStructureTypeDeviceQueueCreateInfo,
      queueFamilyIndex: queueFamilyIndex,
      queueCount: 1'u32,
      pQueuePriorities: addr queuePriority)  # shouldn't this be an UncheckedArray[cfloat]

   var enabledExtensionNames = newCStringArray([vkKhrSwapchainExtensionName.cstring])
   var deviceCreateInfo = VkDeviceCreateInfo(
      sType: vkStructureTypeDeviceCreateInfo,
      queueCreateInfoCount: 1,
      pQueueCreateInfos: addr deviceQueueCreateInfo,
      enabledExtensionCount: enabledExtensionNames.len.uint32,
      ppEnabledExtensionNames: enabledExtensionNames.impl)

   doAssert vkCreateDevice(physicalDevice, addr deviceCreateInfo, nil, addr device) == vkSuccess
   vkGetDeviceQueue(device, queueFamilyIndex, 0, addr queue)

   var
      event: Event
      running = true

   while running:
      while pollEvent(event):
         if event.kind == QuitEvent:
            running = false


   vkDestroySurfaceKHR(instance, surface, nil)
   destroy(window)
   vkDestroyInstance(instance, nil)
   sdl2.quit()

main()
