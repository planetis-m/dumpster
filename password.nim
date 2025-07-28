import std/[sysrand, strutils, assertions, sequtils, oserrors]

type
  CharacterClass = enum
    ccUppercase, ccLowercase, ccDigits, ccSpecials

  PasswordOptions = object
    length: int                                 # Length of password
    charClasses: set[CharacterClass]            # Which character classes to include
    minRequirements: array[CharacterClass, int] # Minimum counts for each class
    excludedChars: set[char]                    # Characters to exclude
    specialChars: set[char]                     # Defaults to punctuation chars

proc newPasswordOptions*(
    length: Positive = 16,
    requirements: openarray[(CharacterClass, int)] = [],
    excludedChars: set[char] = {},
    specialChars: set[char] = PunctuationChars
  ): PasswordOptions =
  ## Creates a new PasswordOptions object with specified settings
  var charClasses: set[CharacterClass] = {}
  var minRequirements {.noinit.}: array[CharacterClass, int]

  # Start with all classes as optional (minCount = 0).
  for c in CharacterClass:
    charClasses.incl(c)
    minRequirements[c] = 0

  for (charClass, minCount) in items(requirements):
    if minCount >= 0:
      # Include and Require
      charClasses.incl(charClass)
    else:
      charClasses.excl(charClass)
    minRequirements[charClass] = minCount

  PasswordOptions(
    length: length,
    charClasses: charClasses,
    minRequirements: minRequirements,
    excludedChars: excludedChars,
    specialChars: specialChars
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
  of ccSpecials: options.specialChars

proc getFullCharSet(options: PasswordOptions): set[char] =
  ## Gets the combined set of all allowed characters based on options
  result = {}
  for charClass in options.charClasses:
    result = result + getCharSet(charClass, options)
  # Remove excluded characters
  result = result - options.excludedChars

proc validateOptions(options: PasswordOptions): bool =
  # First, calculate the total set of characters that are actually available.
  let totalAvailableChars = getFullCharSet(options)
  if totalAvailableChars.len == 0:
    return false
  # Second, calculate the total minimum required characters.
  var totalMinimum = 0
  for charClass in options.charClasses:
    let minCount = options.minRequirements[charClass]
    if minCount > 0:
      totalMinimum += minCount
      # Check if the character pool for this required class is empty.
      let availableCharsInClass = getCharSet(charClass, options) - options.excludedChars
      if availableCharsInClass.len == 0:
        return false # This required class has no characters available.
  # Finally, check if the minimums can be satisfied by the password length.
  if totalMinimum > options.length:
    return false
  return true

proc generateRandomChars(length: int, allowedChars: seq[char]): string =
  ## Generates a password of the specified length using characters from the provided set.
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
  raise newException(ValueError,
    "Failed to generate password meeting constraints after " & $maxAttempts & " attempts.")

proc generateDefaultPassword(): string =
  ## Convenience function to generate a secure password with the default options.
  generatePassword(newPasswordOptions())

when isMainModule:
  echo "Default Password: ", generateDefaultPassword()
  # Paypal only allows: !"#$%&()*+=@\^~
  let paypalSpecials = {'!', '"', '#', '$', '%', '&', '(', ')', '*', '+', '=', '@', '\\', '^', '~'}
  echo "PayPal-Compliant Password: ", generatePassword(newPasswordOptions(requirements = {ccSpecials: 1}, specialChars = paypalSpecials))
  try:
    let customOpts = newPasswordOptions(20, {ccUppercase: 5, ccLowercase: 5, ccDigits: 5, ccSpecials: -1})
    echo "Custom Password: ", generatePassword(customOpts)
  except ValueError as e:
    echo "Caught expected error: ", e.msg
