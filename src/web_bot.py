import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from src.selectors import SELECTORS
from src.config import WEB_URL, WEB_USERNAME, WEB_PASSWORD
from src.logger import get_logger

logger = get_logger(__name__)


class WebBot:
    def __init__(self, url=WEB_URL, username=WEB_USERNAME, password=WEB_PASSWORD, dry_run=False):
        self.url = url
        self.username = username
        self.password = password
        self.dry_run = dry_run
        self.driver = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self):
        """Launches the Chrome browser"""
        if self.dry_run:
            logger.info("[Dry Run] Skipping browser start.")
            return True

        logger.info("Starting Chrome browser.")
        try:
            self.driver = webdriver.Chrome()
            self.driver.maximize_window()
            return True
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            raise

    def login(self):
        """Logs into the web application"""
        if self.dry_run:
            logger.info(f"[Dry Run] Logging in as {self.username}")
            return True

        logger.info(f"Logging into {self.url} as {self.username}")
        self.driver.get(self.url)
        wait = WebDriverWait(self.driver, 10)

        try:
            username_field = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, SELECTORS['login']['text']))
            )
            username_field.send_keys(self.username)

            password_field = self.driver.find_element(By.CSS_SELECTOR, SELECTORS['login']['password'])
            password_field.send_keys(self.password)

            submit_btn = self.driver.find_element(By.CSS_SELECTOR, SELECTORS['login']['submit'])
            submit_btn.click()

            time.sleep(1)
            logger.info("Successfully logged in.")
            return True
        except Exception as e:
            logger.error(f"Login failed: {e}")
            raise

    def navigate_to_table(self):
        """Navigates to the payment entry table"""
        if self.dry_run:
            logger.info("[Dry Run] Navigating to table.")
            return True

        logger.info("Navigating to table.")
        wait = WebDriverWait(self.driver, 15)

        try:
            # 1) Switch to "base" iframe
            iframe = wait.until(EC.presence_of_element_located((By.ID, "base")))
            self.driver.switch_to.frame(iframe)

            # 2) Click on KPSoftPro subtree
            el = wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    "//div[contains(@class,'subtree') and contains(., 'KPSoftPro')]"
                ))
            )
            el.click()

            # 3) Click menu item 3
            el = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, SELECTORS['menu']['stavka_3'])))
            el.click()

            # 4) Click sub-item 16
            el = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, SELECTORS['menu']['podstavka_16'])))
            el.click()

            # 5) Handle nested iframes
            iframe_client = wait.until(EC.presence_of_element_located((By.ID, "undefined")))
            self.driver.switch_to.frame(iframe_client)

            iframe_menu = wait.until(EC.presence_of_element_located((By.ID, "menu_ifrm")))
            self.driver.switch_to.frame(iframe_menu)

            # 6) Click table cell/button
            el = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, SELECTORS['menu']['tabela_celija'])))
            el.click()

            # 7) Final iframe sequence to reach the document popup
            self.driver.switch_to.default_content()
            iframe_base = self.driver.find_element(By.ID, "base")
            self.driver.switch_to.frame(iframe_base)

            iframe_client = self.driver.find_element(By.ID, "undefined")
            self.driver.switch_to.frame(iframe_client)

            iframe_popup = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.ID, "popupDocument_iframe"))
            )
            self.driver.switch_to.frame(iframe_popup)

            logger.info("Successfully navigated to the payment table.")
            return True
        except Exception as e:
            logger.error(f"Navigation to table failed: {e}")
            raise

    def enter_payment(self, ime, iznos):
        """Enters a single payment into the table"""
        if self.dry_run:
            logger.info(f"[Dry Run] Entering payment: {ime} - {iznos}")
            return True

        logger.debug(f"Entering payment: {ime} - {iznos}")
        try:
            # 1) Add new row
            add_buttons = self.driver.find_elements(
                By.CSS_SELECTOR, "tfoot input[type='button'][value='+']"
            )
            add_buttons[-1].click()
            time.sleep(0.15)

            # 2) TAB navigation - Fetching active element fresh after each TAB
            el = self.driver.switch_to.active_element
            el.send_keys(Keys.TAB)
            time.sleep(0.05)

            # TAB 2 (npr. JIB)
            el = self.driver.switch_to.active_element
            el.send_keys(Keys.TAB)
            time.sleep(0.05)

            # TAB 3 – now should be the name field (kupacNaziv)
            el = self.driver.switch_to.active_element
            el.send_keys(Keys.TAB)
            time.sleep(0.05)

            # 3) Enter name into the now active field
            el = self.driver.switch_to.active_element
            el.send_keys(ime)
            time.sleep(0.7)  # Wait for autocomplete

            # 4) Select from autocomplete
            try:
                wait = WebDriverWait(self.driver, 5)
                star_btn = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "#choose_div table#listTable tr#hrow_1 input#opt_1"))
                )
                star_btn.click()  # Select the first customer

                # 5) TAB to the amount column
                el = self.driver.switch_to.active_element
                el.send_keys(iznos)
                time.sleep(0.15)

                logger.debug(f"Successfully entered payment for {ime}")
                return True
            except TimeoutException:
                logger.warning(f"Autocomplete not found for: {ime}")
                # Even if autocomplete fails, we TAB to amount and enter it
                el = self.driver.switch_to.active_element
                el.send_keys(Keys.TAB)
                time.sleep(0.05)
                el = self.driver.switch_to.active_element
                el.send_keys(iznos)
                time.sleep(0.15)
                return False

        except Exception as e:
            logger.error(f"Error entering payment for {ime}: {e}")
            return False

    def save(self) -> bool:
        """Clicks the save button and waits for confirmation"""
        if self.dry_run:
            logger.info("[Dry Run] Saving changes.")
            return True

        logger.info("Saving document.")
        try:
            save_btn = self.driver.find_element(By.CSS_SELECTOR, SELECTORS['actions']['save'])
            save_btn.click()

            # Wait up to 10 seconds for save to complete
            # For this app, we wait 1 second after click
            time.sleep(1)
            logger.info("Save clicked.")
            return True
        except Exception as e:
            logger.error(f"Failed to click save: {e}")
            return False

    def _clear_active_field(self):
        """Clears currently focused input field value."""
        if self.dry_run:
            logger.info("[Dry Run] Clearing active field.")
            return True
        try:
            el = self.driver.switch_to.active_element
            el.send_keys(Keys.CONTROL, "a")
            el.send_keys(Keys.DELETE)
            time.sleep(0.1)
            return True
        except Exception as e:
            logger.warning(f"Failed to clear active field: {e}")
            return False

    def retry_name_in_current_row(self, ime, iznos) -> bool:
        """
        Retry customer resolution in the already opened row.
        Does not click "+" or create a new row.
        """
        if self.dry_run:
            logger.info(f"[Dry Run] Retry in current row: {ime} - {iznos}")
            return True

        try:
            wait = WebDriverWait(self.driver, 5)
            name_inputs = self.driver.find_elements(
                By.CSS_SELECTOR,
                "tbody tr td input[name='kupacNaziv'], tbody tr td input.kupac-naziv",
            )
            if not name_inputs:
                name_inputs = self.driver.find_elements(
                    By.XPATH,
                    "//tbody//tr[last()]//input[@type='text']",
                )

            if not name_inputs:
                logger.warning("Cannot find name field in current row for retry.")
                return False

            last_name_input = name_inputs[-1]
            last_name_input.click()
            time.sleep(0.1)
            last_name_input.send_keys(Keys.CONTROL, "a")
            last_name_input.send_keys(Keys.DELETE)
            time.sleep(0.1)

            last_name_input.send_keys(ime)
            time.sleep(0.7)

            try:
                star_btn = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "#choose_div table#listTable tr#hrow_1 input#opt_1"))
                )
                star_btn.click()
                el = self.driver.switch_to.active_element
                el.send_keys(iznos)
                time.sleep(0.15)
                logger.debug(f"Retry successful for {ime}")
                return True
            except TimeoutException:
                logger.warning(f"Retry autocomplete not found for: {ime}")
                return False
        except Exception as e:
            logger.error(f"Error in retry_name_in_current_row for {ime}: {e}")
            return False

    def _delete_current_empty_row(self):
        """Deletes the last unfinished row from the table."""
        if self.dry_run:
            logger.info("[Dry Run] Deleting current empty row.")
            return True
        try:
            delete_buttons = self.driver.find_elements(
                By.CSS_SELECTOR,
                "tfoot input[type='button'][value='-']",
            )
            if delete_buttons:
                delete_buttons[-1].click()
                time.sleep(0.1)
            return True
        except Exception as e:
            logger.warning(f"Unable to delete empty row: {e}")
            return False

    def close(self):
        """Closes the Chrome browser"""
        if self.driver:
            logger.info("Closing browser.")
            self.driver.quit()
            self.driver = None