"""Open the deployed Streamlit app in a real (headless) browser so it never goes to sleep.
If it is already asleep, press the wake-up button. Fails (non-zero exit) if the app does not render."""
import re
import sys

from playwright.sync_api import sync_playwright

URL = "https://hw2-mlr-yenkai.streamlit.app/"
MARKER = "King County"  # text shown by problem 1, the default page

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, timeout=120_000)
    page.wait_for_timeout(10_000)
    wake = page.get_by_role("button", name=re.compile("get this app back up", re.I))
    if wake.count():
        print("App was asleep -> waking it up")
        wake.first.click()
        page.wait_for_timeout(5_000)
    for _ in range(36):  # up to ~3 minutes for a cold start
        if any(MARKER in (f.content() or "") for f in page.frames):
            print("App is awake and rendering.")
            browser.close()
            sys.exit(0)
        page.wait_for_timeout(5_000)
    page.screenshot(path="keep_awake_failure.png", full_page=True)
    print("App did not render within the time limit.")
    browser.close()
    sys.exit(1)
