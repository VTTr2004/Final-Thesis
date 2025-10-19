
import time
import json
import requests as rq
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Đọc dữ liệu camera từ file JSON
with open(r"data/DataScource.json", "r", encoding='utf-8') as f:
    data = json.load(f)

# Cấu hình Chrome
chrome_binary = r"C:\Users\1tram\OneDrive\Documents\Chrome_Driver\chrome-win64\chrome.exe"
chromedriver_path = r"C:\Users\1tram\OneDrive\Documents\Chrome_Driver\chromedriver-win64\chromedriver.exe"

chrome_options = Options()
chrome_options.binary_location = chrome_binary

service = Service(executable_path=chromedriver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Danh sách các đoạn đường cần crawl
Streets_train = ["Duong_5", "Duong_6", "Duong_7", "Duong_8", "Duong_9", "Duong_10"]
types = ["train"]
#Streets_test  = ["Duong_1"]
#types = ["train", "test"]

src_url = "C:/Users/1tram/OneDrive/Documents/DoAnTotNghiep/All_Dataset/Dataset_Anotation/images"

total_count = 0

def crawl_data(data, street_type, types, amount, dest):
    global total_count
    try:
        count = 0
        while count <= amount:
            count += 1
            for street in street_type:
                for cam_number in data[types][street].keys():
                    try:
                        # Mở trang web camera
                        driver.get(data[types][street][cam_number]['url'])
                        time.sleep(15)

                        # Lấy cookie từ trình duyệt (để request ảnh sau này)
                        cookies = driver.get_cookies()
                        session = rq.Session()
                        for cookie in cookies:
                            session.cookies.set(cookie['name'], cookie['value'])

                        # Lấy user-agent để tránh server chặn
                        headers = {
                            'User-Agent': driver.execute_script("return navigator.userAgent;")
                        }

                        # Lấy tất cả các thẻ <img>
                        imgs = driver.find_elements(By.TAG_NAME, "img")
                        for img in imgs:
                            img_url = img.get_attribute("src")
                            if img_url:
                                file_name = f"{dest}\\{street}_{cam_number}_{data[types][street][cam_number]['amount'] + 1}.png"

                                # Tải ảnh bằng requests với cookie + header
                                img_data = session.get(img_url, headers=headers).content
                                with open(file_name, "wb") as f:
                                    f.write(img_data)

                                # Cập nhật biến đếm
                                data[types][street][cam_number]['amount'] += 1
                                total_count += 1
                                print(f"Đã lưu ảnh thứ: {total_count}")
                    except Exception as e:
                        print(f"Lỗi khi xử lý camera {cam_number} ở {street}: {e}")
                        continue
    except Exception as e:
        print("Không thể kết nối hoặc lỗi khác:", e)

if __name__ == "__main__":
    dest = r"C:\Users\1tram\OneDrive\Documents\DoAnTotNghiep\All_Dataset\Dataset_Anotation\600imgs"
    
    crawl_data(data, Streets_train, "train", 49, dest)

driver.quit()
