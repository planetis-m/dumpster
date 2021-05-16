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

proc main() =
  let session = createSession(Firefox)
  session.navigate(cuUrl)

  let loginCss = "div.login-button"
  let loginBtn = session.waitForElement(loginCss).get()
  loginBtn.click()

  let nameId = "Username"
  let pwdId = "Password"
  let nameInput = session.findElement(nameId, IDSelector).get()
  let pwdInput = session.findElement(pwdId, IDSelector).get()

  nameInput.sendKeys(username)
  pwdInput.sendKeys(password, Key.Enter)

  let paketaCss = "a[title=\"Ενεργά Πακέτα\"]"
  let paketaBtn = session.waitForElement(paketaCss).get()
  paketaBtn.click()
  echo "Logged in"

  let bundle = Bundle.CuWeekly
  let bundleCss = &"a[data-code=\"{bundle}\"]"
  let bundleBtn = session.waitForElement(bundleCss).get()
  discard session.executeScript("arguments[0].click();", bundleBtn)

  let activationCss = "a.bundleActivationOnPopup"
  let activationBtn = session.waitForElement(activationCss).get()
  activationBtn.click()
  echo "Bundle activated"
  sleep(10_000)

  session.stop()

main()
