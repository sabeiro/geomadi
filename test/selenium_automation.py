# -*- coding: utf-8 -*-
#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/py/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as expected
from selenium.webdriver.support.wait import WebDriverWait

options = Options()
options.add_argument('-headless')
driver = Firefox(executable_path='geckodriver', firefox_options=options)
wait = WebDriverWait(driver, timeout=10)
driver.get('https://www.google.com/maps/place/Sch%C3%B6nhauser+Allee+Arcaden/@52.5432723,13.4004447,15z/data=!4m5!3m4!1s0x47a85200ffffffff:0xa5239c459b2d413f!8m2!3d52.5496296!4d13.4153607')
wait.until(expected.visibility_of_element_located((By.NAME, 'q'))).send_keys('headless firefox' + Keys.ENTER)
wait.until(expected.visibility_of_element_located((By.CSS_SELECTOR, '#ires a'))).click()
with open(baseDir + "raw/roda/page.html","w") as f:
    f.write(driver.page_source)

driver.quit()

fs = pd.read_csv(baseDir + "gis/roda/cilac_geom.csv")


def test_sel_webdriver_new_user(self):
    driver = self.driver
    HOST = "myhost.mycompany.com"
    RANDINT = random.random()*10000
    driver.get("https://" + HOST)
    driver.find_element_by_xpath("//*[@id=’nav’]/ol/li[3]/a").click()
    driver.find_element_by_xpath("//*[@id=’top’]/body/div/div[2]/div[2]/div/div[2]/ul/li[4]/a/span").click()
    driver.find_element_by_xpath("//*[@id=’product-collection-image-374']").click()
    driver.find_element_by_xpath("//*[@id=’checkout-button’]/span/span").click()

    driver.find_element_by_id("billing:firstname").clear()
    driver.find_element_by_id("billing:firstname").send_keys("selenium", RANDINT, "_fname")

    driver.find_element_by_id("billing:lastname").clear()
    driver.find_element_by_id("billing:lastname").send_keys("selenium", RANDINT, "_lname")

    # Click Place Order
    driver.find_element_by_xpath("//*[@id=’order_submit_button’]").click()

def is_element_present(self, how, what):
    try: self.driver.find_element(by=how, value=what)
    except NoSuchElementException as e: return False
    return True

def is_alert_present(self):
    try: self.driver.switch_to_alert()
    except NoAlertPresentException as e: return False
    return True

def close_alert_and_get_its_text(self):
    try:
        alert = self.driver.switch_to_alert()
        alert_text = alert.text
        if self.accept_next_alert:
            alert.accept()
        else:
            alert.dismiss()
        return alert_text
    finally: self.accept_next_alert = True

def tearDown(self):
    self.driver.quit()
    self.assertEqual([], self.verificationErrors)

if __name__ == "__main__":
    unittest.main()
