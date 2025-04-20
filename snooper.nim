import osproc, streams

proc snapshot(
    device = "/dev/video0",
    resolution = "640x480",
    seekTime = "0:0:2",
    outputFile = "snapshot.jpg") =
  # ffmpeg -f video4linux2 -s 640x480 -i /dev/video0 -ss 0:0:2 -frames 1 /tmp/out.jpg
  let a = [
    "-f", "video4linux2", "-s", resolution, "-i", device,
    "-ss", seekTime, "-frames", "1" , outputFile]
  let p = startProcess("ffmpeg", args = a, options = {poEchoCmd, poUsePath, poStdErrToStdOut})

  try:
    discard p.waitForExit()
    if p.hasData():
      let outp = p.outputStream()
      let resp = outp.readAll()
      echo(resp)
  finally:
    p.close()

snapshot()
