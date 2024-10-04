import argparse
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--leaderboard', type=str, required=True, choices=['classic', 'lite', 'mmlu'])
    args = parser.parse_args()
    
    base_url = f'https://crfm.stanford.edu/{args.leaderboard}/classic/latest/#/runs?page='
    output_dir = "../../../data/gather_data/crawl_real"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/crawl_dataset_name_{args.leaderboard}.csv'
    if args.leaderboard == "classic":
        total_pages = 86+1
    elif args.leaderboard == "lite":
        total_pages = 21+1
    elif args.leaderboard == "mmlu":
        total_pages = 36+1
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    failed_pages = []
    for page in tqdm(range(1, total_pages)):
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

            if page == 1:
                df_output = df_new
            else:
                df_exist = pd.read_csv(output_path)
                df_output = pd.concat([df_exist, df_new], ignore_index=True)

            df_output.to_csv(output_path, index=False)

        except Exception as e:
            print(f"Error on page {page}: {e}")
            failed_pages.append(page)

    driver.quit()

    if failed_pages:
        print(f"Failed pages: {failed_pages}")
    else:
        print("All pages scraped successfully.")
