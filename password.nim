import std/[sysrand, strutils, assertions, sequtils, oserrors]

type
  CharacterClass = enum
    ccUppercase, ccLowercase, ccDigits, ccSpecials

  PasswordOptions = object
    length: int                                 # Length of password
    charClasses: set[CharacterClass]            # Which character classes to include
    minRequirements: array[CharacterClass, int] # Minimum counts for each class
    excludedChars: set[char]                    # Characters to exclude
    allowedSpecials: set[char]

proc newPasswordOptions(
    length: Positive = 16,
    includeUppercase = true,
    includeLowercase = true,
    includeDigits = true,
    includeSpecials = true,
    minUppercase = 1,
    minLowercase = 1,
    minDigits = 1,
    minSpecials = 1,
    excludedChars: set[char] = {},
    allowedSpecials: set[char] = PunctuationChars
  ): PasswordOptions =
  ## Creates a new PasswordOptions object with specified settings
  var charClasses: set[CharacterClass] = {}
  var minRequirements = default(array[CharacterClass, int])

  if includeUppercase:
    charClasses.incl(ccUppercase)
    minRequirements[ccUppercase] = minUppercase

  if includeLowercase:
    charClasses.incl(ccLowercase)
    minRequirements[ccLowercase] = minLowercase

  if includeDigits:
    charClasses.incl(ccDigits)
    minRequirements[ccDigits] = minDigits

  if includeSpecials:
    charClasses.incl(ccSpecials)
    minRequirements[ccSpecials] = minSpecials

  PasswordOptions(
    length: length,
    charClasses: charClasses,
    minRequirements: minRequirements,
    excludedChars: excludedChars,
    allowedSpecials: allowedSpecials
  )

proc getCharacterClass(c: char): CharacterClass =
  ## Determines which character class a character belongs to
  case c
  of UppercaseLetters: ccUppercase
  of LowercaseLetters: ccLowercase
  of Digits: ccDigits
  else: ccSpecials

proc getCharSet(charClass: CharacterClass, options: PasswordOptions): set[char] =
  ## Returns the set of characters for a given character class
  case charClass
  of ccUppercase: UppercaseLetters
  of ccLowercase: LowercaseLetters
  of ccDigits: Digits
  of ccSpecials: options.allowedSpecials

proc getFullCharSet(options: PasswordOptions): set[char] =
  ## Gets the combined set of all allowed characters based on options
  result = {}
  for charClass in options.charClasses:
    result = result + getCharSet(charClass, options)
  # Remove excluded characters
  result = result - options.excludedChars

proc validateOptions(options: PasswordOptions): bool =
  ## Validates that password options are consistent and possible to satisfy.
  if options.charClasses.card == 0:
    return false
  var totalMinimum = 0
  var totalAvailableChars: set[char] = {}
  for charClass in options.charClasses:
    let minCount = options.minRequirements[charClass]
    totalMinimum += minCount
    # Check if each required class has enough available characters
    let baseChars = getCharSet(charClass, options)
    let availableCharsInClass = baseChars - options.excludedChars
    if availableCharsInClass.card == 0 and minCount > 0:
      # Impossible to satisfy minimum for this class as the pool is empty.
      return false
    # Add the unique available characters to the total pool
    totalAvailableChars = totalAvailableChars + availableCharsInClass
  # Check if minimum requirements exceed total length
  if totalMinimum > options.length:
    return false
  # Check if there are any characters left to generate the password from
  if totalAvailableChars.card == 0:
    return false
  return true

proc generateRandomChars(length: int, allowedChars: seq[char]): string =
  ## Generates a password of the specified length using characters from the provided set.
  assert allowedChars.len > 0, "Character set cannot be empty"
  result = newString(length)
  let numAllowed = allowedChars.len
  let usableRange = (256 div numAllowed) * numAllowed
  var charCount = 0
  # Buffer for random bytes, fetched in chunks to minimize syscalls.
  var randomBytes = newSeq[byte](length * 2)
  var byteIndex = randomBytes.len
  while charCount < length:
    # If the buffer is exhausted, fetch a new chunk.
    if byteIndex >= randomBytes.len:
      if not urandom(randomBytes):
        raiseOSError(osLastError())
      byteIndex = 0
    let randValue = ord(randomBytes[byteIndex])
    inc byteIndex
    # Rejection sampling: only use bytes within the calculated uniform range.
    if randValue < usableRange:
      result[charCount] = allowedChars[randValue mod numAllowed]
      inc charCount

proc generatePassword(options: PasswordOptions): string =
  ## Generates a random password according to the given options
  assert validateOptions(options), "Invalid password options"

  let fullCharSet = getFullCharSet(options)
  let allowedChars = toSeq(fullCharSet)
  var attempt = 0
  const maxAttempts = 100 # Prevent infinite loops for impossible constraints
  while attempt < maxAttempts:
    inc attempt
    result = generateRandomChars(options.length, allowedChars)
    var classCounts = default(array[CharacterClass, int])
    block requirementCheck:
      for ch in result.items:
        let charClass = getCharacterClass(ch)
        inc classCounts[charClass]
      # Verify all minimums are met
      for charClass in options.charClasses:
        if classCounts[charClass] < options.minRequirements[charClass]:
          break requirementCheck
      # # Shuffle the result before returning
      # shuffle(result)
      return
  # If we exit the loop, we failed to meet constraints after many tries
  quit "Failed to generate password meeting constraints after " & $maxAttempts & " attempts."

proc generateDefaultPassword(): string =
  ## Convenience function to generate a secure password with the default options.
  generatePassword(newPasswordOptions())

when isMainModule:
  # Paypal only allows: !"#$%&()*+=@\^~
  let paypalSpecials = {'!', '"', '#', '$', '%', '&', '(', ')', '*', '+', '=', '@', '\\', '^', '~'}
  echo generatePassword(newPasswordOptions(allowedSpecials = paypalSpecials))
  # echo generateDefaultPassword()
