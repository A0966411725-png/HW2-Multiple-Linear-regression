"""Capture website screenshots for README / report (run while `streamlit run app.py` is serving on :8501).
    python .github/scripts/take_screenshots.py [base_url]
"""
import os
import sys

from playwright.sync_api import sync_playwright

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "screenshots")
os.makedirs(OUT, exist_ok=True)
TABS = [("🏠 估價", "1_predict"), ("📊 資料探索", "1_eda"), ("🧪 特徵選擇", "1_feature_selection"),
        ("📈 模型評估", "1_evaluation"), ("📄 報告與 AI 輔助", "1_report")]


def wait_ready(page):
    page.wait_for_timeout(2500)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120_000)
    page.wait_for_timeout(2500)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1400, "height": 1700}, device_scale_factor=1.5)
    page.goto(f"{BASE_URL}/?problem=1", timeout=120_000)
    wait_ready(page)
    for label, name in TABS:
        page.get_by_role("tab", name=label).click()
        wait_ready(page)
        page.screenshot(path=os.path.join(OUT, f"{name}.png"))
        print("saved", name)
    for q, name in [(2, "2_startups"), (3, "3_custom_dataset")]:
        page.goto(f"{BASE_URL}/?problem={q}", timeout=120_000)
        wait_ready(page)
        page.screenshot(path=os.path.join(OUT, f"{name}.png"))
        print("saved", name)
    browser.close()
