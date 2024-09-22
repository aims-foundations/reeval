from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
url = "https://crfm.stanford.edu/helm/classic/latest/#/groups/boolq"
driver.get(url)
rows = driver.find_elements(By.TAG_NAME, 'tr')

wait = WebDriverWait(driver, 20)
wait.until(EC.presence_of_element_located((By.TAG_NAME, 'tr')))
rows = driver.find_elements(By.TAG_NAME, 'tr')
rows = rows[1:]

data = []
for row in rows:
    try:
        # Extract model name from the 'a' tag first
        model_element = row.find_element(By.CSS_SELECTOR, 'a.link.link-hover')
        model_name = model_element.get_attribute('title').split(':')[-1].strip()

        # Check if the model name is empty
        if model_name == "":
            
            raise ValueError("model name is empty")

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
            print("Model name not found")
    
    try:
        score_element = row.find_element(By.CSS_SELECTOR, 'div.flex.items-center')
        score = score_element.text.strip()
        if score == "":
            raise ValueError("score is empty")
    
    except (Exception, ValueError) as e:
        score = ""
        print("Score not found")
                
    data.append([model_name, score])

driver.quit()
df = pd.DataFrame(data, columns=['model_name', 'score'])
df.to_csv('model_scores.csv', index=False)