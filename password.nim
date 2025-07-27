import std/[sysrand, strutils, assertions]

type
  CharacterClass = enum
    ccUppercase, ccLowercase, ccDigits, ccSpecials

  PasswordOptions = object
    length: int                                 # Length of password
    charClasses: set[CharacterClass]            # Which character classes to include
    minRequirements: array[CharacterClass, int] # Minimum counts for each class
    excludedChars: set[char]                    # Characters to exclude

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
    excludedChars: set[char] = {}
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
    excludedChars: excludedChars
  )

proc validateOptions(options: PasswordOptions): bool =
  ## Validates that password options are consistent and possible to satisfy
  result = true
  # Check if we have any character classes to work with
  if options.charClasses.card == 0:
    return false
  # Check if minimum requirements exceed total length
  var totalMinimum = 0
  for charClass, minCount in options.minRequirements.pairs:
    if charClass notin options.charClasses:
      # Can't satisfy minimum for a class that's not included
      return false
    totalMinimum += minCount
  if totalMinimum > options.length:
    return false

proc generateRandomChars(length: int, charsSet: set[char]): string =
  ## Generates a password of the specified length using characters from the provided set,
  assert charsSet.len > 0, "Character set cannot be empty"
  # Initialize result string
  result = newStringOfCap(length) # Reserve capacity for efficiency
  while result.len < length:
    # How many more chars we need?
    let needed = length - result.len
    # Fetch slightly more random bytes than needed (factor of ~3 seems reasonable)
    let bytesToFetch = max(needed, length) * 3
    let randomBytes = urandom(bytesToFetch)

    for byteVal in randomBytes:
      # Only consider bytes within the valid ASCII range (0-127)
      if byteVal <= 127:
        let c = char(byteVal)
        if c in charsSet: # Check if the character is in our allowed set
          result.add(c)
          if result.len == length:
            break # Stop adding once we reach the desired length

proc getCharacterClass(c: char): CharacterClass =
  ## Determines which character class a character belongs to
  case c
  of UppercaseLetters: ccUppercase
  of LowercaseLetters: ccLowercase
  of Digits: ccDigits
  else: ccSpecials

proc getCharSet(charClass: CharacterClass): set[char] =
  ## Returns the set of characters for a given character class
  case charClass
  of ccUppercase: UppercaseLetters
  of ccLowercase: LowercaseLetters
  of ccDigits: Digits
  of ccSpecials: PunctuationChars

proc getFullCharSet(options: PasswordOptions): set[char] =
  ## Gets the combined set of all allowed characters based on options
  result = {}
  for charClass in options.charClasses:
    result = result + getCharSet(charClass)
  # Remove excluded characters
  result = result - options.excludedChars

proc generatePassword(options: PasswordOptions): string =
  ## Generates a random password according to the given options
  assert validateOptions(options), "Invalid password options"

  let fullCharSet = getFullCharSet(options)
  var attempt = 0
  const maxAttempts = 100 # Prevent infinite loops for impossible constraints
  while attempt < maxAttempts:
    inc attempt
    result = generateRandomChars(options.length, fullCharSet)
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
      return result
  # If we exit the loop, we failed to meet constraints after many tries
  quit "Failed to generate password meeting constraints after " & $maxAttempts & " attempts."

proc generateDefaultPassword(): string =
  ## Convenience function to generate a secure password with the default options.
  # Paypal only allows: !"#$%&()*+=@\^~
  # generatePassword(newPasswordOptions(excludedChars={'\'', ','..'/', ':'..'<', '>', '?', '[', ']', '_', '`', '{'..'}'}))
  generatePassword(newPasswordOptions())

when isMainModule:
  echo generateDefaultPassword()
