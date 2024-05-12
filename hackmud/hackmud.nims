task deploy, "Build for the game":
  exec "nim js -d:danger hackmud.nim"
  exec "closure-compiler hackmud.js --js_output_file hackmud_min.js"
