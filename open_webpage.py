from playwright.sync_api import sync_playwright

def main():
    """
    Open the CryptoPanic webpage using Playwright and hold it for 20 seconds.
    """
    url = ""

    with sync_playwright() as p:
        # Launch Chromium browser in headed mode (show browser window)
        browser = p.chromium.launch(headless=False)

        try:
            # Create a new page
            page = browser.new_page()

            # Navigate to the URL with a 10-second timeout
            print(f"Navigating to: {url}")
            page.goto(url, timeout=10000)

            print("Page loaded successfully. Holding for 20 seconds...")

            # Hold for 20 seconds using Playwright's wait_for_timeout
            page.wait_for_timeout(20000)

            print("20 seconds completed. Closing browser.")

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Always close the browser
            browser.close()

if __name__ == "__main__":
    main()