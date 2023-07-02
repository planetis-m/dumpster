import std / [options, os, strformat]
import halonium

const
  cuUrl = "https://www.vodafonecu.gr/"

  username = ""
  password = ""

type
  Bundle {.pure.} = enum
    # Voice
    TalkText600 = "BDLTalkText600" # 10€
    ComboMax = "BDLComboMax" # 12€
    CuComboXL = "BDLCUComboXL" # 15€
    VoiceDay = "BDLVoiceDay" # 1.5€
    CuWeekly = "BDLCUWeekly" # 3€
    ComboVideo1 = "BDLComboVideo1" # 8.5€
    WkCombo = "BDLwkCombo" # 5€
    ComboSocial1 = "BDLComboSocial1" # 8.5€

let selectedBundle = Bundle.CuWeekly # The bundle that will be activated.

proc main() =
  # Program that navigates to the Vodafone CU website, logs in with the provided credentials,
  # activates a specific bundle, and stops the session.

  let session = createSession(Firefox)
  session.navigate(cuUrl)

  let loginCss = "div.login-button"
  let loginBtn = session.waitForElement(loginCss).get()
  loginBtn.click()

  let nameId = "Username"
  let nameInput = session.findElement(nameId, IDSelector).get()
  nameInput.sendKeys(username)

  let pwdId = "Password"
  let pwdInput = session.findElement(pwdId, IDSelector).get()
  pwdInput.sendKeys(password, Key.Enter)

  let packagesCss = "a[title=\"Ενεργά Πακέτα\"]"
  let packagesBtn = session.waitForElement(packagesCss).get()
  packagesBtn.click()

  echo "Logged in"

  let bundleCss = &"a[data-code=\"{selectedBundle}\"]"
  let bundleBtn = session.waitForElement(bundleCss).get()
  discard session.executeScript("arguments[0].click();", bundleBtn)

  let activationCss = "a.bundleActivationOnPopup"
  let activationBtn = session.waitForElement(activationCss).get()
  activationBtn.click()

  echo "Bundle activated"
  sleep(10_000)
  session.stop()

main()
