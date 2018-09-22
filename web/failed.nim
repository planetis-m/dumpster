import asyncdispatch, httpclient, strutils, sets

{.reorder: on.}

const
   startUrl = "http://www.ethnos.gr/"
   searchWord = "stemming"
   maxPagesToVisit = 42

var
   pagesVisited = initSet[string]()
   numPagesVisited = 0
   pagesToVisit = @[startUrl]

proc main =
   while numPagesVisited < maxPagesToVisit:
      waitFor crawl()
   echo("Reached max limit of number of pages to visit.")

proc crawl(): {.async.} =
   var tasks: array[3, Future[void]]
   for i in 0 .. high(tasks):

      var nextPage = pagesToVisit.pop()
      while nextPage in pagesVisited and pagesToVisit.len > 0:
         # We've already visited this page
         nextPage = pagesToVisit.pop()

      tasks[i] = visitPage(nextPage)
   await all(tasks)

proc visitPage(url: string) {.async.} =
   # Add page to our set
   pagesVisited.incl(url)
   numPagesVisited.inc

   let client = newAsyncHttpClient()
   # Make the request
   echo("Visiting page ", url)
   var content: string
   try:
      content = await client.getContent(url)
      if searchWord in content:
         echo("Word ", searchWord, " found at page ", url)
      else:
         for u in content.getUrls:
            pagesToVisit.add(u)
   except:
      echo("Error while visting ", url)


proc crawl() =
   var tasks: array[3, Future[void]]
   while numPagesVisited < maxPagesToVisit:
      for i in 0 .. high(tasks):
         if pagesToVisit.len == 0: break
         var nextPage = pagesToVisit.pop()
         while nextPage in pagesVisited and pagesToVisit.len > 0:
            # We've already visited this page
            nextPage = pagesToVisit.pop()
         tasks[i] = visitPage(nextPage)
      waitFor all(tasks)
