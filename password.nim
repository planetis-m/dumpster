# https://research.kudelskisecurity.com/2020/07/28/the-definitive-guide-to-modulo-bias-and-how-to-avoid-it/
import std/[sysrand, strutils, sequtils, oserrors]

# ==================
# Type Definitions
# ==================

type
  CharacterClass = enum
    ccUppercase, ccLowercase, ccDigits, ccSpecials

  PasswordOptions = object
    length: int                                 # Length of password
    charClasses: set[CharacterClass]            # Which character classes to include
    minRequirements: array[CharacterClass, int] # Minimum counts for each class
    excludedChars: set[char]                    # Characters to exclude
    specialChars: set[char]                     # Defaults to punctuation chars

  PasswordGenerationError = object of CatchableError
    # Raised when password generation fails to meet constraints after multiple attempts

# ==================
# Configuration Constants
# ==================

const
  MaxPasswordAttempts = 100 # Maximum attempts for password generation

# ==================
# Password Options Creation
# ==================

proc newPasswordOptions(
    length: Positive = 16,
    requirements: openarray[(CharacterClass, int)] = [],
    excludedChars: set[char] = {},
    specialChars: set[char] = PunctuationChars): PasswordOptions =
  ## Creates a new PasswordOptions object with specified settings
  ## Parameters:
  ##   - length: Password length (default: 16)
  ##   - requirements: Character class requirements (positive = required, negative = excluded)
  ##   - excludedChars: Characters to exclude from generation
  ##   - specialChars: Special character set to use (default: punctuation)

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

# ==================
# Utility Functions
# ==================

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
    result.incl getCharSet(charClass, options)
  # Remove excluded characters
  result.excl options.excludedChars

# ==================
# Validation Logic
# ==================

proc validateOptions(options: PasswordOptions): bool =
  ## Validates password options to ensure they're feasible
  ## Returns false if:
  ##   - No characters are available
  ##   - Required character classes have no available characters
  ##   - Minimum requirements exceed password length

  # Check if any characters are available
  let totalAvailableChars = getFullCharSet(options)
  if totalAvailableChars.len == 0:
    return false
  # Calculate total minimum required characters
  var totalMinimum = 0
  for charClass in options.charClasses:
    let minCount = options.minRequirements[charClass]
    if minCount > 0:
      totalMinimum += minCount
      # Check if the character pool for this required class is empty.
      let availableCharsInClass = getCharSet(charClass, options) - options.excludedChars
      if availableCharsInClass.len == 0:
        return false # This required class has no characters available.
  # Check if minimum requirements fit in password length
  if totalMinimum > options.length:
    return false
  return true

# ==================
# Password Generation
# ==================

proc generateRandomChars(length: int, allowedChars: seq[char]): string =
  ## Generates a password of the specified length using characters from the provided set.
  ## Uses rejection sampling to avoid modulo bias.

  result = newString(length)
  let numAllowed = allowedChars.len
  # Calculate maximum value that avoids modulo bias
  let usableRange = (256 div numAllowed) * numAllowed
  var charCount = 0
  # Buffer for random bytes, fetched in chunks to minimize syscalls.
  var randomBytes = newSeq[byte](length + length div 2)
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

# ==================
# Core Functions
# ==================

proc generatePassword(options: PasswordOptions): string =
  ## Generates a random password according to the given options
  if not validateOptions(options):
    raise newException(ValueError, "Invalid password options: " & $options)

  let allowedChars = toSeq(getFullCharSet(options))
  var attempt = 0
  while attempt < MaxPasswordAttempts:
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
      return result
  # If we exit the loop, we failed to meet constraints after many tries
  raise newException(PasswordGenerationError,
    "Failed to generate password meeting constraints after " & $MaxPasswordAttempts & " attempts.")

proc generateDefaultPassword(): string =
  ## Convenience function to generate a secure password with the default options.
  generatePassword(newPasswordOptions())

# ==================
# Main Execution Block
# ==================

when isMainModule:
  echo "Default Password: ", generateDefaultPassword()

  # Paypal only allows: !"#$%&()*+=@\^~
  let paypalSpecials = {'!', '"', '#', '$', '%', '&', '(', ')', '*', '+', '=', '@', '\\', '^', '~'}
  echo "PayPal-Compliant Password: ", generatePassword(newPasswordOptions(requirements = {ccSpecials: 1}, specialChars = paypalSpecials))

  try:
    let customOpts = newPasswordOptions(20, {ccUppercase: 3, ccLowercase: 3, ccDigits: 3, ccSpecials: 1}, specialChars = paypalSpecials)
    echo "Custom Password: ", generatePassword(customOpts)
  except PasswordGenerationError:
    discard
