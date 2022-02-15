from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver
from decimal import Decimal
import pandas as pd
import time
import re

class Scrapper_Translate():
    def __init__(self, text, browser='Chrome'):

        self.text = text
        self.LOGIN_URL = "https://translate.google.com/"


        self.chrome_options = webdriver.ChromeOptions()
        prefs = {"profile.default_content_setting_values.notifications": 2}
        self.chrome_options.add_experimental_option("prefs", prefs)
        # self.chrome_options.add_argument("--headless")

        if browser == 'Chrome':
            self.driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(),
                                           chrome_options=self.chrome_options)
        elif browser == 'Firefox':
            self.driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
        self.driver.get(self.LOGIN_URL)
        time.sleep(1)

    def Translate(self):
        text = ""
        try:
            gauche = WebDriverWait(self.driver, 100).until(EC.presence_of_element_located((By.XPATH, "//div[@class='sl-more tlid-open-source-language-list']")))
            self.driver.implicitly_wait(10)
            ActionChains(self.driver).move_to_element(gauche).click(gauche).perform()
            time.sleep(1)

            languages = WebDriverWait(self.driver, 100).until(EC.presence_of_element_located((By.XPATH, "//div[@class='language-list-unfiltered-langs-sl_list']/div[3]")))
            name_language = languages.find_element_by_xpath(".//div[22]")
            self.driver.implicitly_wait(10)
            ActionChains(self.driver).move_to_element(name_language).click(name_language).perform()
            time.sleep(1)

            text_source = WebDriverWait(self.driver, 100).until(EC.presence_of_element_located((By.XPATH, "//textarea[@id='source']")))
            text_source.send_keys(self.text)

            ###################


            droit = WebDriverWait(self.driver, 100).until(EC.presence_of_element_located((By.XPATH, "//div[@class='tl-more tlid-open-target-language-list']")))
            self.driver.implicitly_wait(10)
            ActionChains(self.driver).move_to_element(droit).click(droit).perform()
            time.sleep(1)

            languages = WebDriverWait(self.driver, 100).until(EC.presence_of_element_located((By.XPATH, "//div[@class='language-list-unfiltered-langs-tl_list']/div[2]")))
            name_language = languages.find_element_by_xpath(".//div[28]")
            print(name_language.text)
            self.driver.implicitly_wait(10)
            ActionChains(self.driver).move_to_element(name_language).click(name_language).perform()
            time.sleep(8)
            text_dist = WebDriverWait(self.driver, 100).until(EC.presence_of_element_located((By.XPATH, "//div[@class='text-wrap tlid-copy-target']")))
            text = text_dist.text
            self.driver.close()

        except:
            print("error in loading!")

        return text


