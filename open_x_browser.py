#!/usr/bin/env python3
"""
Simple X.com Browser Opener

This script opens a browser window to x.com and keeps it open
so you can manually test logging into your account.

Usage:
    poetry run python open_x_browser.py

The browser will stay open until you close it manually.
"""

import os
import sys
import time
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Open a browser window to x.com for manual login testing."""

    print("🌐 Opening X.com Browser for Manual Login Testing")
    print("=" * 60)
    print("This script will:")
    print("1. Open a browser window to x.com")
    print("2. Keep it open for manual login testing")
    print("3. Wait for you to close the browser manually")
    print()
    print("Instructions:")
    print("- Use the browser window to test manual login")
    print("- Close the browser window when you're done")
    print("- Press Ctrl+C in this terminal to exit early")
    print()

    try:
        # Start Playwright
        playwright = sync_playwright().start()

        # Launch browser with stealth settings
        browser = playwright.chromium.launch(
            headless=False,  # Show browser window
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-default-apps',
                '--disable-popup-blocking'
            ]
        )

        # Create browser context
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1280, 'height': 800},
            ignore_https_errors=True
        )

        # Create new page
        page = context.new_page()

        # Add stealth script to avoid bot detection
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });

            // Override the permissions API
            if (navigator.permissions) {
                navigator.permissions.query = () => Promise.resolve({ state: 'granted' });
            }

            // Override the plugins API
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });

            // Override the languages API
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
        """)

        # Navigate to x.com
        print("🚀 Opening browser window...")
        page.goto("https://x.com", wait_until='domcontentloaded')

        print("✅ Browser opened successfully!")
        print("🌐 URL: https://x.com")
        print()
        print("Browser window is now open. You can:")
        print("- Test manual login to your account")
        print("- Navigate the site as needed")
        print("- Close the browser window when finished")
        print()
        print("⏳ Waiting for you to close the browser...")

        # Wait until browser is closed
        try:
            # Keep the script running while browser is open
            while True:
                if not browser.is_connected():
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⚠️  Early exit requested")

        print("✅ Browser session completed")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    finally:
        # Clean up resources
        try:
            if 'context' in locals():
                context.close()
            if 'browser' in locals():
                browser.close()
            if 'playwright' in locals():
                playwright.stop()
        except:
            pass

    return 0

if __name__ == "__main__":
    sys.exit(main())