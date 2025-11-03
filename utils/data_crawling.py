import time, json, os
import requests as rq
from io import BytesIO
from PIL import Image

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException

from concurrent.futures import ThreadPoolExecutor, as_completed

class TraficImagesCrawler:
    def __init__(self, chrome_binary_path : str, chrome_driver_path : str):
        self.data = None
        self.chrome_binary_path = chrome_binary_path
        self.chrome_driver_path = chrome_driver_path
        
    def init_driver(self, ) -> None:
        chrome_options = Options()
        chrome_options.binary_location = self.chrome_binary_path
        
        driver = webdriver.Chrome(
            service = Service(executable_path = self.chrome_driver_path), 
            options = chrome_options
        )
        return driver
    
    def load_source(self, source_path : str) -> None:
        with open(source_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            return
        
    def get_dataitems(self):
        all_items = {}
        for type_name, type_value in self.data.items():
            roads = []
            # all_items["types"].append(type_name)
            for road in type_value.keys():
                roads.append(road)
            all_items[type_name] = roads
        print(all_items)
        
    def ensure_dotpng(self, img_data : bytes) -> bytes:
        try:
            img = Image.open(BytesIO(img_data))
            output = BytesIO()
            img.save(output, format="PNG")
            return output.getvalue()
        
        except Exception as e:
            print(f"Image convert png failed: {str(e)}")
            return None
        
    def crawling(self, url : str, save_path, subfix, max_items : int = 10):
        driver = None
        count = 0
        try:
            driver = self.init_driver()
            driver.get(url)
            time.sleep(15)
            
            cookies = driver.get_cookies()
            session = rq.Session()
            for cookie in cookies:
                session.cookies.set(cookie['name'], cookie['value'])
                    
            headers = {
                'User-Agent': driver.execute_script("return navigator.userAgent;")
            }
            
            while count < max_items:
                img = driver.find_element(By.TAG_NAME, "img")
                
                img_url = img.get_attribute("src")
                if not img_url:
                    time.sleep(8)
                    continue
                
                try:
                    img_data = session.get(img_url, headers=headers, timeout=8).content
                    png_data = self.ensure_dotpng(img_data)
                except:
                    print("Skip invalid image.")
                    time.sleep(8)
                    continue
                
                if png_data:
                    filename = os.path.join(save_path, f"{subfix}_{count}.png")
                    with open(filename, 'wb') as f:
                        f.write(png_data)
                    print(f"Saved: {filename}")
                    count += 1
                time.sleep(8)
                
            print(f"*** Total saved: {count}")

                
        except WebDriverException as wde:
            print(f"Failed (Selenium/Network): {str(wde)}")
            time.sleep(5)
        except rq.exceptions.RequestException as re:
            print(f"Failed (Request Download): {str(re)}")
            time.sleep(5)
        except Exception as e:
            print(f"Failed (General): {str(e)}")
            time.sleep(5)
        finally:
            if driver:
                driver.quit()
            
    def crawling_handle_All(self, save_folder_path : str, max_items_per_obj = 5):
        for type_name, types_data in self.data.items():
            desc_folder = os.path.join(save_folder_path, type_name)
            os.makedirs(desc_folder, exist_ok=True)
            
            for road_id, roads_data in types_data.items():
                for cam_id, cams_data in roads_data.items():
                    subfix = f"{road_id}_{cam_id}"
                    url = cams_data["url"]
                    self.crawling(url, save_path=desc_folder, subfix=subfix, max_items=max_items_per_obj)  
                    
    def crawling_handle_multithread(self, save_folder_path, max_items_per_obj=5, workers=3):
        tasks = []
        for type_name, types_data in self.data.items():
            folder = os.path.join(save_folder_path, type_name)
            os.makedirs(folder, exist_ok=True)

            for road_id, roads_data in types_data.items():
                for cam_id, cam_info in roads_data.items():
                    tasks.append((
                        cam_info["url"],
                        folder,
                        f"{road_id}_{cam_id}",
                        max_items_per_obj
                    ))

        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = [exe.submit(self.crawling, *t) for t in tasks]
            for f in as_completed(futures):
                f.result()
                    
if __name__ == "__main__":
    chrome_binary = r"C:\Users\1tram\OneDrive\Documents\Chrome_Driver\chrome-win64\chrome.exe"
    chromedriver_path = r"C:\Users\1tram\OneDrive\Documents\Chrome_Driver\chromedriver-win64\chromedriver.exe"

    crawler = TraficImagesCrawler(
        chrome_binary_path = chrome_binary,
        chrome_driver_path = chromedriver_path
    )
    
    crawler.load_source(r"data\DataScource.json")
    
    crawler.crawling_handle_multithread("tmp/test_crawl_data", 5, workers=2)