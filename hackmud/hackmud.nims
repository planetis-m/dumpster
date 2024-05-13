task up, "Build for the game":
  exec "nim js -d:danger t1_cracker.nim"
  exec "closure-compiler t1_cracker.js --js_output_file t1_cracker_min.js"
