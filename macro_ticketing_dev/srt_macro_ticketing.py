'''
Script by Jin Hyun Park @ Texas A&M University
Automated SRT ticketing system
Warning: Automated scripts for purchasing a ticket is illegal (so ... this is an illegal code). Do not share!

How to use:
There are only two parts that you need to change, noted by arrow (<----------------)
1. replace id field by user membership number
2. replace password field by user password number
'''

from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.alert import Alert

import time

driver = webdriver.Chrome()

# Go to the SRT reservation website
driver.get('https://etk.srail.kr/main.do')

# Start logging in
login_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[text()="로그인"]')))
login_button.click()

# Id
id_field = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="srchDvNm01"]')))
id_field.send_keys('1692956545')    # <----------------

# Password
pwd_field = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="hmpgPwdCphd01"]')))
pwd_field.send_keys('11bv60636063*')    # <----------------

# Finish logging in
submit_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@type="submit"]')))
submit_button.click()

# departure
departure_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="dptRsStnCd"]')))
departure_button.click()
station_dropdown = WebDriverWait(driver, 2).until(
    EC.presence_of_element_located((By.XPATH, '//*[@id="dptRsStnCd"]')))
select = Select(station_dropdown)
select.select_by_visible_text('수서')

# destination
destination_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, '//*[@id="arvRsStnCd"]')))
destination_button.click()
station_dropdown = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, '//*[@id="arvRsStnCd"]')))
select = Select(station_dropdown)
select.select_by_visible_text('동대구')

# time
time_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="dptTm"]')))
time_button.click()
station_dropdown = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="dptTm"]')))
select = Select(station_dropdown)
select.select_by_visible_text('18시 이후')

# start search!
search_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, '//*[@onclick[contains(., "selectScheduleList()")]]')))
search_button.click()

# --------------- 여기부터 예약가능한 리스트가 나온다. ---------------
while True:

    driver.refresh()

    try:
        # 예약 상황1: '입석+좌석' 승차권 예약
        button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, '//a/span[text()="입석+좌석"]')))
    except TimeoutException:
        # 예약 상황2: '예약하기' 승차권 예약
        button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, '//a/span[text()="예약하기"]')))

    button.click()

    try:
        # Wait for the alert to appear
        WebDriverWait(driver, 3).until(EC.alert_is_present())

        # Switch to the alert
        alert = Alert(driver)

        # Accept the alert (click on the OK button)
        alert.accept()

    except TimeoutException:
        print("No alert present.")
        pass

    try:
        # Try to find and click the '확인' button
        button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, '//a/span[text()="확인"]')))
        button.click()

    except NoSuchElementException:
        # If '확인' button is not found, find and click the '결제하기' button
        button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, '//a/span[text()="결제하기"]')))
        button.click()
        break




time.sleep(864000)  # 10 days



# buy_button = WebDriverWait(driver, 10).until(
#     EC.element_to_be_clickable((By.XPATH, '//*[@onclick="settleAmount();"]')))
# buy_button.click()