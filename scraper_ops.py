import requests
import os
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# ✅ เพิ่ม Facebook Graph API config
FACEBOOK_ACCESS_TOKEN = "EAANIKWNNZCZBIBQ6KoHxPi42dz2vD5qx7vkOjkZBd5CIUwpO3srzGdW29QGCLc0nqnsjClZAh83khHJmielICgUQ2drgx6ukzGkbCYHIbPT8KT8US8bqseUphTg4gtlbeS39fw9nJTmL1oPW5OJ7CfqOCWs4dMbhzKTFrLwaqepgYuRXk88n04BixO9ZCO8bMOnscAalWP7fIlkNE3BYcpXIS4jTbjV6Y9Lc4MFiJc35IMvIJcHZB4TXdwYdRQ1kQ7KuHKtuKaRzH3OLkZD"  # ใส่ token ที่ได้จาก Graph API Explorer
def refresh_facebook_token():
    """Auto refresh token ก่อนหมดอายุ"""
    token     = os.getenv("FACEBOOK_ACCESS_TOKEN", "")
    app_id    = os.getenv("FACEBOOK_APP_ID", "")
    app_secret = os.getenv("FACEBOOK_APP_SECRET", "")

    if not all([token, app_id, app_secret]):
        return token

    try:
        url = "https://graph.facebook.com/v18.0/oauth/access_token"
        params = {
            "grant_type":        "fb_exchange_token",
            "client_id":         app_id,
            "client_secret":     app_secret,
            "fb_exchange_token": token
        }
        res  = requests.get(url, params=params, timeout=10)
        data = res.json()

        if "access_token" in data:
            new_token = data["access_token"]
            # ✅ อัปเดต .env อัตโนมัติ
            _update_env("FACEBOOK_ACCESS_TOKEN", new_token)
            return new_token
    except Exception as e:
        print(f"⚠️ Token refresh failed: {e}")

    return token


def _update_env(key: str, value: str):
    """อัปเดตค่าใน .env file"""
    env_path = ".env"
    try:
        with open(env_path, "r") as f:
            lines = f.readlines()
        with open(env_path, "w") as f:
            updated = False
            for line in lines:
                if line.startswith(f"{key}="):
                    f.write(f"{key}={value}\n")
                    updated = True
                else:
                    f.write(line)
            if not updated:
                f.write(f"{key}={value}\n")
    except Exception as e:
        print(f"⚠️ Cannot update .env: {e}")

def get_facebook_post(url: str):
    """ดึงข้อมูลจาก Facebook post/page ผ่าน Graph API"""
    try:
        # ดึง post_id จาก URL
        # รองรับรูปแบบ: facebook.com/page/posts/123 หรือ facebook.com/permalink.php?story_fbid=123
        import re
        post_id = None

        patterns = [
            r'facebook\.com/.+/posts/(\d+)',
            r'facebook\.com/permalink\.php\?story_fbid=(\d+)',
            r'facebook\.com/photo\?fbid=(\d+)',
            r'/(\d{10,})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                post_id = match.group(1)
                break

        if not post_id:
            return None, "⚠️ ไม่สามารถดึง Post ID จาก URL นี้ได้"

        if not FACEBOOK_ACCESS_TOKEN:
            return None, "⚠️ ยังไม่ได้ตั้งค่า Facebook Access Token"

        # เรียก Graph API
        api_url = f"https://graph.facebook.com/v18.0/{post_id}"
        params = {
            "fields": "message,story,full_picture,created_time",
            "access_token": FACEBOOK_ACCESS_TOKEN
        }
        res = requests.get(api_url, params=params, timeout=15)
        data = res.json()

        if "error" in data:
            return None, f"⚠️ Facebook API Error: {data['error'].get('message','Unknown error')}"

        message = data.get("message") or data.get("story") or ""
        if not message:
            return None, "⚠️ ไม่พบข้อความใน Post นี้ (อาจเป็น Private)"

        created_time = data.get("created_time", "")
        title = f"Facebook Post ({created_time[:10] if created_time else 'ไม่ทราบวันที่'})"
        return title, message

    except Exception as e:
        return None, f"⚠️ Facebook Error: {str(e)}"


def get_content_from_url(url):
    # ✅ ดักทาง Facebook แยกออกมาใช้ Graph API
    if "facebook.com" in url.lower():
        return get_facebook_post(url)

    # ปิด social media อื่นๆ
    blocked_domains = ["x.com", "twitter.com", "tiktok.com", "instagram.com"]
    if any(domain in url.lower() for domain in blocked_domains):
        return None, "⚠️ ไม่สามารถดึงข้อมูลจาก Social Media ได้โดยตรง กรุณาก๊อปปี้ข้อความมาวางในโหมด 'พิมพ์เนื้อหา' แทนครับ"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        title = None
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"]
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True)
        else:
            title = "ไม่พบหัวข้อข่าว"

        article_body = soup.find("article")
        if article_body:
            paragraphs = article_body.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        content_list = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
        full_content = "\n".join(content_list)

        if not full_content:
            return title, "⚠️ ดึงเนื้อหาไม่สำเร็จ: เว็บไซต์นี้อาจใช้ JavaScript ในการแสดงผล หรือบล็อกการดึงข้อมูล"

        return title, full_content

    except requests.exceptions.RequestException as e:
        return None, f"⚠️ ไม่สามารถเชื่อมต่อกับเว็บไซต์ได้\nรายละเอียด: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"