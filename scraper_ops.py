import requests
from bs4 import BeautifulSoup

def get_content_from_url(url):
    # ใส่ User-Agent ปลอมตัวเป็นเบราว์เซอร์ปกติ เพื่อลดโอกาสโดนเว็บไซต์บล็อก
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        # ดึงข้อมูล HTML จาก URL (ตั้ง Timeout 60 วินาทีให้เหมือนของเดิม)
        response = requests.get(url, headers=headers, timeout=60)
        
        # เช็คว่าโหลดสำเร็จไหม (Status 200) ถ้าติด 404, 500 จะเด้งเข้า Exception
        response.raise_for_status()
        
        # แปลงเนื้อหา HTML ด้วย BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # เช็คว่ามี h1 ไหม ถ้าไม่มีให้ข้ามไป
        title_element = soup.find("h1")
        title = title_element.get_text(strip=True) if title_element else "ไม่พบหัวข้อข่าว"
        
        # ดึงข้อมูลแท็ก p ทั้งหมด
        paragraphs = soup.find_all("p")
        content_list = [p.get_text(strip=True) for p in paragraphs]
        full_content = "\n".join(content_list)
        
        return title, full_content
        
    except Exception as e:
        # หากเกิด Error ใดๆ คืนค่ากลับไปแบบเดิม
        return None, f"Error: {str(e)}"