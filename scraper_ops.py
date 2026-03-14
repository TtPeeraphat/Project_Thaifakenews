# scraper_ops.py
import os
from playwright.sync_api import sync_playwright

os.system("playwright install chromium")

def get_content_from_url(url):
    # ใช้โหมด Sync เพื่อไม่ให้ตีกับ Streamlit บน Windows
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            # เช็คว่ามี h1 ไหม ถ้าไม่มีให้ข้ามไป
            title_element = page.query_selector("h1")
            title = title_element.inner_text() if title_element else "ไม่พบหัวข้อข่าว"
            
            paragraphs = page.query_selector_all("p")
            content_list = [p.inner_text() for p in paragraphs]
            full_content = "\n".join(content_list)
            
            browser.close()
            return title, full_content
            
        except Exception as e:
            browser.close()
            return None, f"Error: {str(e)}"