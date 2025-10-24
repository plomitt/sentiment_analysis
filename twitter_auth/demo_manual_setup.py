#!/usr/bin/env python3
"""
Manual Authentication Setup Demo

This script demonstrates how to set up Twitter/X authentication manually
to bypass bot detection and create a reusable browser state.

Usage:
    1. Run this script with headless=False
    2. Complete any manual verification steps in the browser
    3. The script will save the authentication state automatically
    4. Use the saved state with the main twitter_auth module
"""

import os
import time
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
AUTH_DIR = "twitter_auth/.auth"
STATE_FILE = f"{AUTH_DIR}/twitter_state.json"
TARGET_URL = "https://x.com/search?q=bitcoin&src=typed_query&f=live"


def manual_auth_setup():
    """
    Perform manual authentication setup to capture browser state.

    This opens a browser window where you can complete the login manually,
    including any verification steps that Twitter requires.
    """

    # Ensure auth directory exists
    os.makedirs(AUTH_DIR, exist_ok=True)

    print("🔧 Twitter/X Manual Authentication Setup")
    print("=" * 50)
    print("\nThis script will:")
    print("1. Open a browser window to Twitter/X login")
    print("2. Wait for you to complete the login manually")
    print("3. Save the authentication state for future use")
    print("4. Test access to the Bitcoin search page")
    print("\n⚠️  Important:")
    print("- The browser window will stay open for 5 minutes")
    print("- Complete any verification steps Twitter requires")
    print("- Keep the browser window open until the script finishes")
    print("\nPress Enter to continue...")
    input()

    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(
        headless=False,  # Show the browser window
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

    context = browser.new_context(
        user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        viewport={'width': 1280, 'height': 800},
        ignore_https_errors=True
    )

    page = context.new_page()

    # Add stealth script
    page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });

        if (navigator.permissions) {
            navigator.permissions.query = () => Promise.resolve({ state: 'granted' });
        }

        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });

        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
    """)

    try:
        # Navigate to Twitter login
        print(f"\n🌐 Opening Twitter login page...")
        page.goto("https://x.com/i/flow/login", wait_until='domcontentloaded')

        print("\n📝 Instructions:")
        print("1. Complete the login process in the browser window")
        print("2. Handle any verification steps Twitter shows")
        print("3. Make sure you're fully logged in")
        print("4. The script will check for successful login every 30 seconds")
        print("\n⏱️  Waiting for manual login completion...")

        # Wait for manual login completion
        max_wait_time = 300  # 5 minutes
        check_interval = 30  # Check every 30 seconds

        for elapsed in range(0, max_wait_time, check_interval):
            print(f"   Checking login status... ({elapsed + check_interval}s/{max_wait_time}s)")

            # Check if we're no longer on login pages
            current_url = page.url
            if ('login' not in current_url.lower() and
                'i/flow' not in current_url and
                'home' in current_url or 'search' in current_url):
                print("✅ Login detected! Saving authentication state...")
                break

            # Check for successful login indicators
            try:
                # Look for signs of being logged in
                if page.locator('[data-testid="SideNav_AccountSwitcher_Button"]').count() > 0:
                    print("✅ Login detected via navigation elements! Saving authentication state...")
                    break
            except:
                pass

            if elapsed + check_interval >= max_wait_time:
                print("⏰ Timeout reached. Please try again.")
                browser.close()
                playwright.stop()
                return

            time.sleep(check_interval)

        # Save the authentication state
        print("💾 Saving browser authentication state...")
        state = context.storage_state()

        with open(STATE_FILE, 'w') as f:
            import json
            json.dump(state, f, indent=2)

        print(f"✅ Authentication state saved to: {STATE_FILE}")

        # Test the saved state by accessing the target URL
        print(f"\n🧪 Testing access to Bitcoin search page...")
        page.goto(TARGET_URL, wait_until='domcontentloaded')
        time.sleep(3)

        # Check for success indicators
        tweet_indicators = [
            '[data-testid="tweet"]',
            'article[role="article"]',
            '[data-testid="tweetText"]'
        ]

        found_tweets = False
        for indicator in tweet_indicators:
            if page.locator(indicator).count() > 0:
                found_tweets = True
                break

        if found_tweets:
            print("✅ SUCCESS! Access to authenticated content confirmed!")
            print(f"📊 Current URL: {page.url}")
        else:
            print("⚠️  Limited access detected. This may be normal depending on Twitter's current state.")
            print(f"📊 Current URL: {page.url}")

        print("\n🎉 Setup complete! You can now use the authentication system.")
        print("\nNext steps:")
        print("1. Use: from twitter_auth import get_authenticated_search_results")
        print("2. Call: content = get_authenticated_search_results()")
        print("3. The saved state will be loaded automatically")

        # Keep browser open for a bit to see the result
        print("\n🌐 Browser window will remain open for 10 seconds for inspection...")
        time.sleep(10)

    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")

    finally:
        # Cleanup
        context.close()
        browser.close()
        playwright.stop()
        print("\n🧹 Browser closed. Setup completed.")


if __name__ == "__main__":
    manual_auth_setup()