import futhark

# Tell futhark where to find the C libraries you will compile with, and what
# header files you wish to import.
importc:
  sysPath "/usr/lib/clang/13.0.1/include/"
  path "include"
  "raylib.h"

  retype structmodelanimation.frameposes, ptr UncheckedArray[ptr UncheckedArray[Transform]]

# Tell Nim how to compile against the library. If you have a dynamic library
# this would simply be a `--passL:"-l<library name>`
{.passC: "-Iinclude".}
{.passC: "-D_DEFAULT_SOURCE".}
{.passL: "include/libraylib.a".}
{.passL: "-lX11 -lGL -lm -lpthread -ldl -lrt".}

const
  LIGHTGRAY = Color(r: 200, g: 200, b: 200, a: 255)
  RAYWHITE = Color(r: 245, g: 245, b: 245, a: 255)

# Use the library just like you would in C!
proc main =
  InitWindow(800, 450, "raylib [core] example - basic window")
  while not WindowShouldClose():
    BeginDrawing()
    ClearBackground(RAYWHITE)
    DrawText("Congrats! You created your first window!", 190, 200, 20, LIGHTGRAY)
    EndDrawing()
  CloseWindow()

main()
