import os
from datetime import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Get the current year and month
current_date = datetime.now()
year_month = current_date.strftime("%Y_%m")  # Format as 'YYYY_MM'

# Define the output file path with the desired format
output_file = f"pe_ratios{year_month}.csv"

# Check if the CSV file already exists
if os.path.exists(output_file):
    print(f"'{output_file}' exists. Reading data from the file...")
    # Read the data from the existing CSV file
    df = pd.read_csv(output_file)
else:
    print(f"'{output_file}' does not exist. Launching browser to scrape data...")
    # URL of the webpage
    url = "https://www.macrotrends.net/stocks/charts/NVDA/nvidia/pe-ratio"

    try:
        # Initialize the Chrome WebDriver
        driver = webdriver.Chrome()
        driver.get(url)

        # Wait until the table element is present
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "table")))

        # Find the table by its class name
        table = driver.find_element(By.CLASS_NAME, "table")

        # Extract the HTML content of the table
        table_html = table.get_attribute("outerHTML")

    finally:
        # Close the WebDriver
        driver.quit()

    # Parse the HTML
    soup = BeautifulSoup(table_html, 'html.parser')

    # Extract rows from the table
    rows = soup.find_all('tr')[2:]  # Skip header rows

    # Create lists for years and P/E ratios
    years = []
    pe_ratios = []

    # Loop through rows to extract year and P/E ratio
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 4:
            date = cells[0].get_text(strip=True)  # First cell: Date
            pe_ratio = cells[-1].get_text(strip=True)  # Last cell: P/E Ratio
            year = date.split('-')[0]  # Extract year from date
            years.append(year)
            pe_ratios.append(pe_ratio)

    # Create a DataFrame
    data = {'Year': years, 'P/E Ratio': pe_ratios}
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file with the timestamped filename
    df.to_csv(output_file, index=False)
    print(f"Data scraped and saved locally as '{output_file}'")

# Display the DataFrame
print(df)
