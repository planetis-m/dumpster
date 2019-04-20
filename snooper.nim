import osproc, streams

proc snapshot =
   let a = ["-f", "video4linux2", "-s 640x480", "-i /dev/video0", "-ss 0:0:2", "-frames 1" , "/tmp/out.jpg"]
   let p = startProcess("ffmpeg", args = a, options = {poEchoCmd, poUsePath, poStdErrToStdOut})

   discard p.waitForExit()
   if p.hasData():
      let outp = p.outputStream()
      let resp = outp.readAll()
      echo(resp)
   p.close()

snapshot()
