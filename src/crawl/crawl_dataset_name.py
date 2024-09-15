import argparse
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    args = parser.parse_args()
    exp = args.exp
    
    if exp == "classic":
        base_url = f'https://crfm.stanford.edu/helm/classic/latest/#/runs?q=&page='
        total_pages = 86+1
        csv_file = f'../../data/real/crawl/crawl_dataset_name_classic.csv'
    elif exp == "lite":
        base_url = f'https://crfm.stanford.edu/helm/lite/latest/#/runs?page='
        total_pages = 21+1
        csv_file = f'../../data/real/crawl/crawl_dataset_name_lite.csv'
    elif exp == "mmlu":
        base_url = f'https://crfm.stanford.edu/helm/mmlu/latest/#/runs?page='
        total_pages = 36+1
        csv_file = f'../../data/real/crawl/crawl_dataset_name_mmlu.csv'
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        
    failed_pages = []
    for page in range(1, total_pages):
        try:
            url = f'{base_url}{page}'
            driver.get(url)
            driver.refresh()

            WebDriverWait(driver, 120).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'table.table tbody tr'))
            )

            runs = []
            table_rows = driver.find_elements(By.CSS_SELECTOR, 'table.table tbody tr')
            for row in table_rows:
                run = row.find_elements(By.TAG_NAME, 'td')[0].text.strip()
                runs.append(run)

            df_new = pd.DataFrame(runs, columns=['Run'])

            if os.path.exists(csv_file):
                df_existing = pd.read_csv(csv_file)
                df_updated = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_updated = df_new

            df_updated.to_csv(csv_file, index=False)
            print(f"Page {page} scraped and saved.")

        except Exception as e:
            print(f"Error on page {page}: {e}")
            failed_pages.append(page)

    driver.quit()

    if failed_pages:
        print(f"Failed pages: {failed_pages}")
    else:
        print("All pages scraped successfully.")
