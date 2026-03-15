import requests
from bs4 import BeautifulSoup

def get_content_from_url(url):
    # 1. ดักทาง Social Media ที่ดึงด้วย requests ไม่ได้แน่ๆ
    blocked_domains = ["facebook.com", "x.com", "twitter.com", "tiktok.com", "instagram.com"]
    if any(domain in url.lower() for domain in blocked_domains):
        return None, "⚠️ ไม่สามารถดึงข้อมูลจาก Social Media ได้โดยตรง กรุณาก๊อปปี้ข้อความมาวางในโหมด 'พิมพ์เนื้อหา' แทนครับ"

    # ใส่ User-Agent ปลอมตัวเป็นเบราว์เซอร์ปกติ
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # -----------------------------------------
        # 2. การหา Title (ฉลาดขึ้น)
        # -----------------------------------------
        title = None
        # ลองหาจาก meta property="og:title" (เว็บข่าวส่วนใหญ่ใช้สำหรับแชร์ลงโซเชียล)
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"]
        # ถ้าไม่มี ลองหา h1
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        # ถ้าไม่มีอีก เอา <title> ของหน้าเว็บมาเลย
        elif soup.title:
            title = soup.title.get_text(strip=True)
        else:
            title = "ไม่พบหัวข้อข่าว"
            
        # -----------------------------------------
        # 3. การหาเนื้อหา Content (ฉลาดขึ้น)
        # -----------------------------------------
        # ลองหาจากแท็ก <article> ก่อน (เว็บข่าวมาตรฐานชอบใช้)
        article_body = soup.find("article")
        if article_body:
            paragraphs = article_body.find_all("p")
        else:
            # ถ้าไม่มี <article> ค่อยกวาดแท็ก <p> ทั้งหน้าเหมือนเดิม
            paragraphs = soup.find_all("p")
            
        content_list = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20] # กรองข้อความสั้นๆ (เช่น เมนู, ปุ่ม) ทิ้ง
        full_content = "\n".join(content_list)
        
        # ถ้าดึงมาแล้วเนื้อหาว่างเปล่า (อาจจะเจอเว็บที่เป็น JavaScript ล้วน)
        if not full_content:
             return title, "⚠️ ดึงเนื้อหาไม่สำเร็จ: เว็บไซต์นี้อาจใช้ JavaScript ในการแสดงผล หรือบล็อกการดึงข้อมูล"
             
        return title, full_content
        
    except requests.exceptions.RequestException as e:
        return None, f"⚠️ ไม่สามารถเชื่อมต่อกับเว็บไซต์ได้ (อาจโดนบล็อกหรือลิงก์เสีย)\nรายละเอียด: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"