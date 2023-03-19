import random

proc sample*[T](population: openarray[T]; k: Natural): seq[T] =
   ## Chooses k unique random elements from a population sequence.
   ## Returns a new list containing elements from the population while
   ## leaving the original population unchanged.  The resulting list is
   ## in selection order so that all sub-slices will also be valid random
   ## samples. This allows raffle winners (the sample) to be partitioned
   ## into grand prize and second place winners (the subslices).
   ## Members of the population need not be hashable or unique. If the
   ## population contains repeats, then each occurrence is a possible
   ## selection in the sample.
   ##
   ## Sampling without replacement entails tracking potential
   ## selections (the pool) in a list.
   let n = len(population)
   if k > n:
      raise newException(ValueError, "Sample can't be larger than population")
   result = newSeq[T](k)
   # An n-length list is smaller than a k-length set
   var pool = newSeq[int](n)
   for i in 0 ..< n:
      pool[i] = i
   for i in 0 ..< k: # invariant:  non-selected at [0,n-i)
      let t = n - i - 1
      let j = rand(t)
      result[i] = population[pool[j]]
      pool[j] = pool[t] # move non-selected item into vacancy

proc sample*(population: Slice[int]; k: Natural): seq[int] =
   let n = len(population)
   if k > n:
      raise newException(ValueError, "Sample can't be larger than population")
   result = newSeq[int](k)
   # An n-length list is smaller than a k-length set
   var pool = newSeq[int](n)
   for i in 0 ..< n:
      pool[i] = i + population.a
   for i in 0 ..< k: # invariant:  non-selected at [0,n-i)
      let t = n - i - 1
      let j = rand(t)
      result[i] = pool[j]
      pool[j] = pool[t] # move non-selected item into vacancy

iterator sample*[T](population: openarray[T]; k: Natural): T =
   ## Yields `k` random integers from ``population`` in random order.
   ## Each number has an equal chance to be picked and can be picked only once.
   ##
   ## Raises ``ValueError`` if there are less than `k` items in `population`.
   let n = len(population)
   if k > n:
      raise newException(ValueError, "Sample can't be larger than population")
   # An n-length list is smaller than a k-length set
   var pool = newSeq[int](n)
   for i in 0 ..< n:
      pool[i] = i
   for i in 0 ..< k: # invariant:  non-selected at [0,n-i)
      let t = n - i - 1
      let j = rand(t)
      yield population[pool[j]]
      pool[j] = pool[t] # move non-selected item into vacancy

iterator sample*(population: Slice[int]; k: Natural): int =
   ## Yields `k` random elements from ``population`` in random order.
   ## Each number has an equal chance to be picked and can be picked only once.
   ##
   ## Raises ``ValueError`` if there are less than `k` items in `population`.
   let n = len(population)
   if k > n:
      raise newException(ValueError, "Sample can't be larger than population")
   # An n-length list is smaller than a k-length set
   var pool = newSeq[int](n)
   for i in 0 ..< n:
      pool[i] = i + population.a
   for i in 0 ..< k: # invariant:  non-selected at [0,n-i)
      let t = n - i - 1
      let j = rand(t)
      yield pool[j]
      pool[j] = pool[t] # move non-selected item into vacancy
