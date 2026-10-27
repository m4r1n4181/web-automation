import sys
import pandas as pd
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent))
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from src.config import WEB_URL, WEB_USERNAME, WEB_PASSWORD
from src.selectors import SELECTORS
from src.logger import get_logger

logger = get_logger(__name__)


class CustomerScraper:
    def __init__(self, total_pages=43):
        self.driver = None
        self.total_pages = total_pages  # total number of pages to go through

    def scrape_customers(self, output_path="customers.csv"):
        """
        Full flow:
        login -> navigate -> scrape all pages -> save to CSV
        """
        try:
            self._open_browser()
            self._login()
            self._navigate_to_customers()

            all_customers = self._scrape_all_pages()

            self._save_to_csv(all_customers, output_path)
            logger.info(f"✅ Scraped {len(all_customers)} customers to {output_path}")
            return all_customers
        except Exception as e:
            logger.error(f"❌ Scraper failed: {e}")
            raise
        finally:
            self._close_browser()

    # -------------------- basic browser steps --------------------

    def _open_browser(self):
        logger.info("🚀 Opening browser...")
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()

    def _login(self):
        logger.info("🔐 Logging in...")
        self.driver.get(WEB_URL)
        wait = WebDriverWait(self.driver, 10)

        username_field = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, SELECTORS['login']['text']))
        )
        username_field.send_keys(WEB_USERNAME)

        password_field = self.driver.find_element(
            By.CSS_SELECTOR, SELECTORS['login']['password']
        )
        password_field.send_keys(WEB_PASSWORD)

        submit_btn = self.driver.find_element(
            By.CSS_SELECTOR, SELECTORS['login']['submit']
        )
        submit_btn.click()

        time.sleep(2)
        logger.info("✅ Login successful")

    def _navigate_to_customers(self):
        logger.info("📂 Navigating to customers section...")
        wait = WebDriverWait(self.driver, 15)

        try:
            iframe = wait.until(EC.presence_of_element_located((By.ID, "base")))
            self.driver.switch_to.frame(iframe)

            el = wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    "//div[contains(@class,'subtree') and contains(., 'KPSoftPro')]"
                ))
            )
            el.click()

            el = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, SELECTORS['menu']['stavka_5'])
                )
            )
            el.click()

            el = wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, SELECTORS['menu']['podstavka_5'])
                )
            )
            el.click()

            self.driver.switch_to.default_content()
            iframe_base = self.driver.find_element(By.ID, "base")
            self.driver.switch_to.frame(iframe_base)

            iframe_client = self.driver.find_element(By.ID, "undefined")
            self.driver.switch_to.frame(iframe_client)

            iframe_popup = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.ID, "menu_ifrm"))
            )
            self.driver.switch_to.frame(iframe_popup)

            time.sleep(2)
            logger.info("✅ Reached customers table")
        except TimeoutException:
            logger.error("❌ Failed to locate customers table")
            raise

    # -------------------- core scraping logic --------------------

    def _scrape_all_pages(self):
        """
        Loop through all pages:
        - read current page
        - click next page
        """
        all_customers = []
        prev_ids = None
        wait = WebDriverWait(self.driver, 10)

        for page in range(1, self.total_pages + 1):
            logger.info(f"📄 Page {page}/{self.total_pages}")

            # 1) read current table
            customers = self._extract_customer_table()

            # stop if the page did not change (we are on the last page)
            page_ids = [c["id"] for c in customers]
            if page > 1 and page_ids == prev_ids:
                logger.info("Page did not change - last page reached.")
                break
            prev_ids = page_ids

            all_customers.extend(customers)

            # 2) stop if this is the last page
            if page == self.total_pages:
                break

            # 3) click next page (or page number)
            try:
                next_btn = wait.until(
                    EC.element_to_be_clickable((
                        By.XPATH,
                        "//img[contains(@src, 'sledecastrana.gif')]"
                    ))
                )
                next_btn.click()

                time.sleep(2)  # give the page time to load
            except TimeoutException:
                logger.warning("⚠️ Next page button not found")
                break

        return all_customers

    def _extract_customer_table(self):
        """
        Reads only the current page table.
        """
        logger.info("📊 Reading customers table (current page)...")
        wait = WebDriverWait(self.driver, 10)

        try:
            table = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table#listTable"))
            )
            rows = table.find_elements(By.TAG_NAME, "tr")

            # header names (th or td in the first row) - used to save ALL columns
            header_cells = rows[0].find_elements(By.XPATH, "./th|./td") if rows else []
            headers = [h.text.strip() for h in header_cells]

            customers = []
            # skip header row, process everything else
            for row in rows[1:]:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 6:  # adjust if table structure changes
                    customer = {
                        "name": cols[2].text.strip(),
                        "municipality": cols[5].text.strip(),
                        "id": cols[1].text.strip()
                    }
                    # all columns, named "col<index>_<header>"
                    for i, c in enumerate(cols):
                        h = headers[i] if i < len(headers) and headers[i] else ""
                        customer[f"col{i}_{h}"] = c.text.strip()
                    customers.append(customer)

            logger.info(f"✅ Extracted {len(customers)} customers from this page")
            return customers

        except Exception as e:
            logger.error(f"❌ Error while reading table: {e}")
            raise

    # -------------------- save & close --------------------

    def _save_to_csv(self, customers, output_path):
        df = pd.DataFrame(customers)
        df.to_csv(output_path, index=False, encoding='utf-8')

    def _close_browser(self):
        if self.driver:
            logger.info("🔒 Closing browser")
            self.driver.quit()
            self.driver = None


if __name__ == "__main__":
    # 500 = gornja granica; scraper staje sam kad nema sljedece stranice
    scraper = CustomerScraper(total_pages=500)
    customers = scraper.scrape_customers(str(SCRIPTS_DIR / "customers_all_pages.csv"))
    print("Total customers:", len(customers))
    print("First 5:", customers[:5])