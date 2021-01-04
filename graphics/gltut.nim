import sdl2, opengl

proc main =
   let
      windowWidth = 1366'i32
      windowHeight = 768'i32

   discard sdl2.init(INIT_EVERYTHING)

   let window = createWindow("SDL/OpenGL Skeleton", SDL_WINDOWPOS_UNDEFINED,
         SDL_WINDOWPOS_UNDEFINED, windowWidth, windowHeight, SDL_WINDOW_OPENGL or SDL_WINDOW_RESIZABLE)

   discard glSetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3)
   discard glSetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3)
   discard glSetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE)
   discard glSetAttribute(SDL_GL_ACCELERATED_VISUAL, 1)

   let context = window.glCreateContext()
   loadExtensions()

   var
      event: Event
      running = true

   while running:
      while pollEvent(event):
         if event.kind == QuitEvent:
            running = false

      glClearColor(1, 0, 0, 1)
      glClear(GL_COLOR_BUFFER_BIT)
      window.glSwapWindow()

   destroy(window)
   glDeleteContext(context)
   sdl2.quit()

main()
