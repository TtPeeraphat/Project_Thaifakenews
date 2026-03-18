import requests
import os
import re
from bs4 import BeautifulSoup
from apify_client import ApifyClient
from dotenv import load_dotenv  # ✅ นำเข้าตัวช่วยอ่านไฟล์ .env
from apify_client import ApifyClient

# ✅ โหลดค่าจากไฟล์ .env เข้าสู่ระบบก่อน
load_dotenv()

# ✅ ดึง API Token ของ Apify จาก .env
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")

# ---------------------------------------------------------
# ฟังก์ชัน: ใช้ Apify ดึงเนื้อหา Facebook
# ---------------------------------------------------------
def get_facebook_post_apify(url: str):
    """ใช้ Apify ดึงข้อมูลโพสต์ Facebook แทน requests ธรรมดา"""
    if not APIFY_API_TOKEN:
        return None, "⚠️ ไม่พบ APIFY_API_TOKEN ในระบบ กรุณาตรวจสอบไฟล์ .env"
        
    try:
        print(f"🚀 กำลังส่งให้ Apify ดึงข้อมูล Facebook จาก: {url}")
        
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
            print(f"✅ ดึงสำเร็จ! ความยาวข้อความ: {len(content)} ตัวอักษร")
            return title, content
        else:
            return title, "⚠️ ดึงได้ แต่ไม่พบข้อความในโพสต์นี้ (อาจมีแค่รูปภาพหรือวิดีโอ)"

    except Exception as e:
        return None, f"⚠️ Apify Error (Facebook): {str(e)}"

# ---------------------------------------------------------
# ฟังก์ชันอัปเดต: ใช้ Apify ดึงเนื้อหา X (Twitter)
# ---------------------------------------------------------
# ---------------------------------------------------------
# ฟังก์ชันอัปเดต: ใช้ Apify ดึงเนื้อหา X (Twitter)
# ---------------------------------------------------------
import re # อย่าลืมเช็คว่ามี import re ด้านบนสุดของไฟล์ด้วยนะครับ (ปกติมีอยู่แล้ว)

# ---------------------------------------------------------
# ฟังก์ชันอัปเดต: ใช้ Apify ดึงเนื้อหา X (Twitter) ด้วยบอทตัวใหม่
# ---------------------------------------------------------
def get_x_post_apify(url: str):
    """ใช้ Apify ดึงข้อมูลโพสต์จาก X (Twitter)"""
    if not APIFY_API_TOKEN:
        return None, "⚠️ ไม่พบ APIFY_API_TOKEN ในระบบ กรุณาตรวจสอบไฟล์ .env"
        
    try:
        # ✅ 1. ลบช่องว่างที่อาจติดมากับ URL (เช่น กรณีก็อปปี้มาผิด)
        clean_url = url.replace(" ", "")
        print(f"🚀 กำลังเตรียมดึงข้อมูล X (Twitter) จาก: {clean_url}")
        
        # ✅ 2. สกัดเอาเฉพาะ Tweet ID (ตัวเลขที่อยู่หลัง /status/) ออกมา
        match = re.search(r"status/(\d+)", clean_url)
        if not match:
            return None, "⚠️ ไม่พบ Tweet ID ในลิงก์ กรุณาก๊อปปี้ลิงก์โพสต์ของ X (Twitter) ให้ครบถ้วน"
        
        tweet_id = match.group(1)
        print(f"🔍 สกัด Tweet ID ได้คือ: {tweet_id} -> กำลังส่งให้ Apify...")

        client = ApifyClient(APIFY_API_TOKEN)

        # ✅ 3. คอนฟิก Input สำหรับบอทตัวใหม่ที่รองรับ API ฟรี (ใช้ tweetIDs แทน startUrls)
        run_input = {
            "tweetIDs": [tweet_id], 
            "maxItems": 1, 
        }

        # ✅ 4. เรียกใช้งาน Actor ตัวใหม่ (Pay-Per-Result ตัดจากเครดิตฟรี)
        run = client.actor("kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest").call(run_input=run_input)

        # ไปดึงผลลัพธ์ (Dataset)
        dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items

        if not dataset_items:
            return None, "⚠️ Apify ไม่พบข้อมูล Twitter (โพสต์อาจถูกลบ หรือบัญชีถูกตั้งเป็นส่วนตัว)"

        post_data = dataset_items[0]

        # ✅ 5. ดักจับเนื้อหาและคนโพสต์ (ตามโครงสร้างข้อมูลของบอทตัวใหม่)
        content = post_data.get("text") or post_data.get("full_text") or ""
        author_info = post_data.get("author", {})
        author = author_info.get("userName") or "Unknown Author"
        
        title = f"X (Twitter) Post by @{author}"

        if content:
            print(f"✅ ดึงสำเร็จ! ความยาวข้อความ: {len(content)} ตัวอักษร")
            return title, content
        else:
            return title, f"⚠️ ดึงได้ แต่ไม่พบข้อความ (ข้อมูลดิบ: {str(post_data)[:100]})"

    except Exception as e:
        return None, f"⚠️ Apify Error (X/Twitter): {str(e)}"
# ---------------------------------------------------------
# ฟังก์ชันหลัก
# ---------------------------------------------------------
def get_content_from_url(url):
    # ✅ 1. เช็คว่าเป็นลิงก์ Facebook หรือไม่
    fb_domains = ["facebook.com", "fb.watch", "fb.com", "fb.me"]
    if any(domain in url.lower() for domain in fb_domains):
        return get_facebook_post_apify(url)

    # ✅ 2. เช็คว่าเป็นลิงก์ X (Twitter) หรือไม่
    x_domains = ["x.com", "twitter.com"]
    if any(domain in url.lower() for domain in x_domains):
        return get_x_post_apify(url)

    # ✅ 3. ปิด social media อื่นที่ยังไม่มีระบบรองรับ
    blocked_domains = ["tiktok.com", "instagram.com"]
    if any(domain in url.lower() for domain in blocked_domains):
        return None, "⚠️ ไม่สามารถดึงข้อมูลจาก Social Media ระบบอื่นได้โดยตรง กรุณาก๊อปปี้ข้อความมาวางแทน"

    # ✅ 4. สำหรับเว็บไซต์ข่าวทั่วไป ใช้โค้ด BeautifulSoup เดิม
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"]
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        else:
            title = "ไม่พบหัวข้อข่าว"

        article_body = soup.find("article")
        paragraphs = article_body.find_all("p") if article_body else soup.find_all("p")
        content_list = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
        full_content = "\n".join(content_list)

        if not full_content:
            return title, "⚠️ ดึงเนื้อหาไม่สำเร็จ: เว็บไซต์นี้อาจใช้ JavaScript หรือบล็อกการดึงข้อมูล"

        return title, full_content

    except requests.exceptions.RequestException as e:
        return None, f"⚠️ ไม่สามารถเชื่อมต่อได้\nรายละเอียด: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"