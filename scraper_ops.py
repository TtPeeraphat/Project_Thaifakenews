import requests
import sys
import os



import re
import logging
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from apify_client import ApifyClient

from text_preprocessor import TextPreprocessor



logger = logging.getLogger(__name__)

#  โหลดค่าจากไฟล์ .env เข้าสู่ระบบก่อน
load_dotenv()

# ดึง API Token ของ Apify จาก .env
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")

# ---------------------------------------------------------
# ฟังก์ชัน: ใช้ Apify ดึงเนื้อหา Facebook
# ---------------------------------------------------------
def get_facebook_post_apify(url: str):
    """ใช้ Apify ดึงข้อมูลโพสต์ Facebook แทน requests ธรรมดา"""
    if not APIFY_API_TOKEN:
        return None, "⚠️ ไม่พบ APIFY_API_TOKEN ในระบบ กรุณาตรวจสอบไฟล์ .env"
        
    try:
        logger.info("Fetching Facebook post: %s", url)
        
        # เริ่มต้นใช้งาน Apify Client
        client = ApifyClient(APIFY_API_TOKEN)

        # ตั้งค่า Input สำหรับ Actor
        run_input = {
            "startUrls": [{"url": url}],
            "resultsLimit": 1, # ดึงแค่ 1 โพสต์ตาม URL ที่ส่งมา
        }

        # เรียกใช้งาน Actor และรอจนกว่าจะเสร็จ
        run = client.actor("apify/facebook-posts-scraper").call(run_input=run_input)

        # ไปดึงผลลัพธ์ (Dataset) ที่ Apify รันได้
        dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items

        if not dataset_items:
            return None, "⚠️ Apify ไม่พบข้อมูล (อาจเป็นโพสต์ส่วนตัว กลุ่มปิด หรือถูกลบไปแล้ว)"

        # แกะข้อมูลเนื้อหาออกมา
        post_data = dataset_items[0]
        
        # ✅ ดักจับ Key หลายๆ แบบที่ Apify อาจจะส่งกลับมา
        content = post_data.get("text") or post_data.get("message") or post_data.get("postText") or ""
        title = "Facebook Post"

        if content:
            logger.info("Facebook fetch success: %d chars", len(content))
            return title, content
        else:
            return title, "⚠️ ดึงได้ แต่ไม่พบข้อความในโพสต์นี้ (อาจมีแค่รูปภาพหรือวิดีโอ)"

    except Exception as e:
        err = str(e)   # ← แปลง exception เป็น string เอง
        if "exceed" in err.lower() or "usage" in err.lower() or "billing" in err.lower():
            return None, "⚠️ เครดิต Apify หมดแล้ว — กรุณาก๊อปปี้เนื้อหามาวางแทน"
        return None, f"⚠️ Apify Error (Facebook): {err}"
       

# ---------------------------------------------------------
# ฟังก์ชันอัปเดต: ใช้ Apify ดึงเนื้อหา X (Twitter)
# ---------------------------------------------------------
def get_x_post_apify(url: str):
    """ใช้ Apify ดึงข้อมูลโพสต์จาก X (Twitter)"""
    if not APIFY_API_TOKEN:
        return None, "⚠️ ไม่พบ APIFY_API_TOKEN ในระบบ กรุณาตรวจสอบไฟล์ .env"
        
    try:
        #  ลบช่องว่างที่อาจติดมากับ URL (เช่น กรณีก็อปปี้มาผิด)
        clean_url = url.replace(" ", "")
        logger.info("Fetching X post: %s", clean_url)
        
        #  สกัดเอาเฉพาะ Tweet ID (ตัวเลขที่อยู่หลัง /status/) ออกมา
        match = re.search(r"status/(\d+)", clean_url)
        if not match:
            return None, "⚠️ ไม่พบ Tweet ID ในลิงก์ กรุณาก๊อปปี้ลิงก์โพสต์ของ X (Twitter) ให้ครบถ้วน"
        
        tweet_id = match.group(1)
        logger.info("Tweet ID: %s", tweet_id)

        client = ApifyClient(APIFY_API_TOKEN)

        #  คอนฟิก Input สำหรับบอทตัวใหม่ที่รองรับ API ฟรี (ใช้ tweetIDs แทน startUrls)
        run_input = {
            "tweetIDs": [tweet_id], 
            "maxItems": 1, 
        }

        #  เรียกใช้งาน Actor ตัวใหม่ (Pay-Per-Result ตัดจากเครดิตฟรี)
        run = client.actor("kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest").call(run_input=run_input)

        # ไปดึงผลลัพธ์ (Dataset)
        dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items

        if not dataset_items:
            return None, "⚠️ Apify ไม่พบข้อมูล Twitter (โพสต์อาจถูกลบ หรือบัญชีถูกตั้งเป็นส่วนตัว)"

        post_data = dataset_items[0]

        #  ดักจับเนื้อหาและคนโพสต์ (ตามโครงสร้างข้อมูลของบอทตัวใหม่)
        content = post_data.get("text") or post_data.get("full_text") or ""
        author_info = post_data.get("author", {})
        author = author_info.get("userName") or "Unknown Author"
        
        title = f"X (Twitter) Post by @{author}"

        if content:
            logger.info("X fetch success: %d chars", len(content))
            return title, content
        else:
            return title, f"⚠️ ดึงได้ แต่ไม่พบข้อความ (ข้อมูลดิบ: {str(post_data)[:100]})"

    except Exception as e:
            err = str(e)   # ← แปลง exception เป็น string เอง
            if "exceed" in err.lower() or "usage" in err.lower() or "billing" in err.lower():
                return None, "⚠️ เครดิต Apify หมดแล้ว — กรุณาก๊อปปี้เนื้อหามาวางแทน"
            return None, f"⚠️ Apify Error (Facebook): {err}"
# ---------------------------------------------------------
# ฟังก์ชันหลัก
# ---------------------------------------------------------
def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text   # ← return อยู่ตรงนี้ถูกต้องแล้ว

def get_content_from_url(url):
    fb_domains = ["facebook.com", "fb.watch", "fb.com", "fb.me"]
    if any(domain in url.lower() for domain in fb_domains):
        return get_facebook_post_apify(url)

    x_domains = ["x.com", "twitter.com"]
    if any(domain in url.lower() for domain in x_domains):
        return get_x_post_apify(url)

    blocked_domains = ["tiktok.com", "instagram.com"]
    if any(domain in url.lower() for domain in blocked_domains):
        return None, "⚠️ ไม่สามารถดึงข้อมูลจาก Social Media ระบบอื่นได้โดยตรง"

    headers = {
        "Accept-Language": "th-TH,th;q=0.9",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        og_title_tag = soup.find("meta", property="og:title")
        if og_title_tag is not None:
            og_content = og_title_tag.get("content")
            title = str(og_content) if og_content else "ไม่พบหัวข้อข่าว"
        else:
            h1_tag = soup.find("h1")
            title = h1_tag.get_text(strip=True) if h1_tag is not None else "ไม่พบหัวข้อข่าว"

        article_body = soup.find("article")
        paragraphs = article_body.find_all("p") if article_body else soup.find_all("p")
        content_list = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
        full_content = "\n".join(content_list)

        if not full_content:
            return title, "⚠️ ดึงเนื้อหาไม่สำเร็จ: เว็บไซต์นี้อาจใช้ JavaScript หรือบล็อกการดึงข้อมูล"

       
        title        = clean_html(title)
        full_content = clean_html(full_content)

        return title, full_content

    except requests.exceptions.RequestException as e:
        return None, f"⚠️ ไม่สามารถเชื่อมต่อได้: {str(e)}"
    except Exception as e:
        err = str(e)
        if "exceed" in err.lower() or "usage" in err.lower() or "billing" in err.lower():
            return None, "⚠️ เครดิต Apify หมดแล้ว — ไม่สามารถดึงข้อมูล X/Twitter ได้\nกรุณาก๊อปปี้เนื้อหาโพสต์มาวางแทน"
        return None, f"⚠️ Apify Error (X): {err}"

    
