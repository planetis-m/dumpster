import asynchttpserver, asyncdispatch, asyncfile, mimetypes, os

proc main =
   let httpServer = newAsyncHttpServer()

   proc reqhandler(req: Request) {.async.} =
      let path = req.url.path
      echo "URL ", path
      var err: HttpCode = Http200
      if '.' in path:
         let contentType = getMimetype(newMimeTypes(), splitFile(path).ext)
         let headers = newHttpHeaders({"Content-Type": contentType})
         var file: AsyncFile
         var data = ""
         try:
            file = openAsync("public" / path, fmRead)
         except:
            err = Http404
         if err != Http404:
            data = await file.readAll()
         await req.respond(err, data, headers)
         if err != Http404:
            file.close()
      else:
         err = Http404
         await req.respond(err, "")

   waitFor httpServer.serve(Port(8000), reqhandler)

main()
