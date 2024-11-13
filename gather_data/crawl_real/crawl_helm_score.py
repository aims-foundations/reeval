import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
from selenium.common.exceptions import TimeoutException

if __name__ == "__main__":
    driver = webdriver.Chrome()
    # url = 'https://crfm.stanford.edu/helm/classic/latest/#/scenarios'
    # driver.get(url)
    # wait = WebDriverWait(driver, 300)
    # wait.until(EC.presence_of_element_located((By.TAG_NAME, 'tr')))
    # rows = driver.find_elements(By.TAG_NAME, 'tr')
    # hrefs = []
    # for row in rows:
    #     links = row.find_elements(By.TAG_NAME, 'a')
    #     for link in links:
    #         href = link.get_attribute('href')
    #         if "#/groups" in href:
    #             hrefs.append(href)
    
    output_dir = "../../../data/gather_data/crawl_real/helm_socre"
    os.makedirs(output_dir, exist_ok=True)
    
    hrefs = ["https://crfm.stanford.edu/helm/mmlu/latest/#/leaderboard"]
    
    for url in tqdm(hrefs):
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 300)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'tr')))
        except TimeoutException:
            print(f"{url} failed to load")
            continue
        
        filename = url.split('/')[-1]
        rows = driver.find_elements(By.TAG_NAME, 'tr')
        rows = rows[1:]

        data = []
        for row in rows:
            true_tag = 0
            try:
                td_element = row.find_element(By.CSS_SELECTOR, 'td.z-0.bg-gray-50, td.z-0.bg-white')
                # Find the <a> tag inside the located <td> element
                model_element = td_element.find_element(By.CSS_SELECTOR, 'a.link.link-hover')
                model_name = model_element.get_attribute('title').split(':')[-1].strip()
                if model_name == "":
                    raise ValueError("model name is empty")
                true_tag = 1

            except (Exception, ValueError) as e:
                try:
                    # If model name not found, extract from the 'div' with specific classes
                    model_element = row.find_element(By.CSS_SELECTOR, 'div.underline.decoration-dashed.decoration-gray-300.z-10')
                    model_name = model_element.text.strip()
                    # Check if the model name is empty
                    if model_name == "":
                        raise ValueError("model name is empty")
                except (Exception, ValueError) as e:
                    model_name = ""
                    print(f"Model name not found, url: {url}")
            
            try:
                td_element = row.find_element(By.CSS_SELECTOR, 'td.z-0.bg-gray-50, td.z-0.bg-white')
                score_element = td_element.find_element(By.CSS_SELECTOR, 'div.flex.items-center')
                score = score_element.text.strip()
                if score == "":
                    raise ValueError("score is empty")
            except (Exception, ValueError) as e:
                score = ""
                print(f"Score not found, model name: {model_name}, url: {url}")
                        
            data.append([model_name, score, true_tag])

        df = pd.DataFrame(data, columns=['model_name', 'score', 'true_model_name'])
        df.to_csv(f'{output_dir}/{filename}.csv', index=False)
    
    driver.quit()