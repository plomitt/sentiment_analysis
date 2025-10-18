"""
Twitter/X Text Extraction Methods Comparison

This script tests different Playwright text extraction methods on a specific X.com post
to determine which method provides the cleanest text extraction for LLM processing.
"""

import os
import re
import time
import traceback
from playwright.sync_api import sync_playwright
from datetime import datetime

# Configuration
TARGET_URL = "https://x.com/MerlijnTrader/status/1979585766515761455"
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_PATH = os.path.join(RESULTS_DIR, "screenshot.png")

class TwitterExtractionComparison:
    def __init__(self):
        self.results = {}

    def setup_browser(self):
        """Initialize browser with appropriate settings"""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=True,  # Set to False for debugging
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
        self.context = self.browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1280, 'height': 800},
            ignore_https_errors=True
        )
        self.page = self.context.new_page()

        # Add stealth measures
        self.page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)

    def cleanup_browser(self):
        """Clean up browser resources"""
        if hasattr(self, 'context'):
            self.context.close()
        if hasattr(self, 'browser'):
            self.browser.close()
        if hasattr(self, 'playwright'):
            self.playwright.stop()

    def load_tweet(self, max_wait_time=60):
        """Load the tweet page and wait for content"""
        try:
            print(f"Loading URL: {TARGET_URL}")

            # Try different wait strategies
            try:
                self.page.goto(TARGET_URL, wait_until='domcontentloaded', timeout=max_wait_time * 1000)
            except:
                # Fallback with less strict waiting
                self.page.goto(TARGET_URL, timeout=max_wait_time * 1000)

            # Wait a bit for dynamic content
            print("Waiting for page content to load...")
            self.page.wait_for_timeout(3000)

            # Try multiple selectors for tweet text
            tweet_selectors = [
                '[data-testid="tweetText"]',
                '[data-testid="tweet"] div[lang]',
                'div[lang] span',
                '.css-1dbjc4n div[lang]',
                'div[lang]',
                'article div[lang]'
            ]

            tweet_element = None
            for selector in tweet_selectors:
                try:
                    elements = self.page.locator(selector)
                    if elements.count() > 0:
                        tweet_element = elements.first
                        print(f"Found tweet with selector: {selector}")
                        break
                except:
                    continue

            if not tweet_element:
                print("No tweet content found with known selectors")
                # Try to find any text content as fallback
                try:
                    # Look for any element with text content
                    all_text_elements = self.page.locator('*:has-text(":")').all()
                    for element in all_text_elements:
                        text = element.text_content()
                        if text and len(text) > 20:  # Assume tweet text is reasonably long
                            tweet_element = element
                            print(f"Found fallback text element with {len(text)} chars")
                            break
                except:
                    pass

            if not tweet_element:
                print("No suitable text content found")
                return None

            # Take screenshot for reference
            try:
                self.page.screenshot(path=SCREENSHOT_PATH, full_page=True)
                print(f"Screenshot saved to: {SCREENSHOT_PATH}")
            except:
                print("Could not save screenshot")

            return tweet_element

        except Exception as e:
            print(f"Error loading tweet: {str(e)}")
            traceback.print_exc()
            return None

    def method1_inner_text(self, tweet_element):
        """Method 1: Using inner_text() - Direct equivalent to Chrome's copy"""
        method_name = "Method 1: inner_text()"
        print(f"\n=== {method_name} ===")

        try:
            start_time = time.time()
            text = tweet_element.inner_text()
            end_time = time.time()

            result = {
                'text': text,
                'success': True,
                'time': end_time - start_time,
                'error': None
            }

            print(f"Success! Length: {len(text)} chars")
            print(f"Time: {result['time']:.2f}s")
            print(f"Text preview: {text[:100]}...")

            return result

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return {
                'text': None,
                'success': False,
                'time': 0,
                'error': error_msg
            }

    def method2_javascript_evaluation(self, tweet_element):
        """Method 2: JavaScript evaluation with custom emoji/hashtag handling"""
        method_name = "Method 2: JavaScript evaluation"
        print(f"\n=== {method_name} ===")

        try:
            start_time = time.time()

            # Advanced JavaScript to handle various content types
            js_code = """
            element => {
                const walker = document.createTreeWalker(
                    element,
                    NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT,
                    null,
                    false
                );

                let text = '';
                let node;
                while (node = walker.nextNode()) {
                    if (node.nodeType === Node.TEXT_NODE) {
                        text += node.textContent;
                    } else if (node.nodeName === 'IMG') {
                        // Handle emoji images - use alt text if available
                        if (node.alt && node.alt.trim()) {
                            text += node.alt;
                        } else {
                            // Fallback: try to get emoji from src or other attributes
                            const src = node.src || '';
                            if (src.includes('emoji')) {
                                text += '[emoji]';
                            }
                        }
                    } else if (node.nodeName === 'A' && node.href) {
                        // Handle links - preserve text content
                        text += node.textContent;
                    }
                }

                // Clean up extra whitespace
                return text.replace(/\\s+/g, ' ').trim();
            }
            """

            text = tweet_element.evaluate(js_code)
            end_time = time.time()

            result = {
                'text': text,
                'success': True,
                'time': end_time - start_time,
                'error': None
            }

            print(f"Success! Length: {len(text)} chars")
            print(f"Time: {result['time']:.2f}s")
            print(f"Text preview: {text[:100]}...")

            return result

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return {
                'text': None,
                'success': False,
                'time': 0,
                'error': error_msg
            }

    def method3_accessibility_tree(self, tweet_element):
        """Method 3: Accessibility tree using aria-label"""
        method_name = "Method 3: Accessibility tree"
        print(f"\n=== {method_name} ===")

        try:
            start_time = time.time()

            # Try multiple accessibility attributes
            accessibility_selectors = [
                'aria-label',
                'title',
                'data-text'
            ]

            text = None
            for attr in accessibility_selectors:
                try:
                    value = tweet_element.get_attribute(attr)
                    if value and value.strip():
                        text = value
                        print(f"Found text using attribute: {attr}")
                        break
                except:
                    continue

            # Fallback to inner_text if no accessibility text found
            if not text:
                text = tweet_element.inner_text()
                print("Fallback to inner_text()")

            end_time = time.time()

            result = {
                'text': text,
                'success': True,
                'time': end_time - start_time,
                'error': None
            }

            print(f"Success! Length: {len(text)} chars")
            print(f"Time: {result['time']:.2f}s")
            print(f"Text preview: {text[:100]}...")

            return result

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return {
                'text': None,
                'success': False,
                'time': 0,
                'error': error_msg
            }

    def method4_text_content_cleanup(self, tweet_element):
        """Method 4: text_content() with regex cleanup"""
        method_name = "Method 4: text_content() with cleanup"
        print(f"\n=== {method_name} ===")

        try:
            start_time = time.time()
            raw_text = tweet_element.text_content()

            # Clean up the text
            clean_text = re.sub(r'\s+', ' ', raw_text)  # Replace multiple whitespace with single space
            clean_text = clean_text.strip()  # Remove leading/trailing whitespace
            clean_text = re.sub(r'\n\s*\n', '\n', clean_text)  # Remove multiple newlines

            end_time = time.time()

            result = {
                'text': clean_text,
                'success': True,
                'time': end_time - start_time,
                'error': None
            }

            print(f"Success! Length: {len(clean_text)} chars")
            print(f"Time: {result['time']:.2f}s")
            print(f"Text preview: {clean_text[:100]}...")

            return result

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return {
                'text': None,
                'success': False,
                'time': 0,
                'error': error_msg
            }

    def save_result_to_file(self, method_name, result):
        """Save method result to a text file"""
        filename = f"method{method_name.split(':')[0].split()[-1]}_{method_name.split(':')[1].strip().lower().replace(' ', '_')}.txt"
        filepath = os.path.join(RESULTS_DIR, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== {method_name} ===\n")
                f.write(f"URL: {TARGET_URL}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Success: {result['success']}\n")
                f.write(f"Time: {result['time']:.2f}s\n")
                f.write(f"Text Length: {len(result['text']) if result['text'] else 0} chars\n")

                if result['error']:
                    f.write(f"Error: {result['error']}\n")

                f.write("\n" + "="*50 + "\n")
                f.write("EXTRACTED TEXT:\n")
                f.write("="*50 + "\n")

                if result['text']:
                    f.write(result['text'])
                else:
                    f.write("[No text extracted]")

            print(f"Result saved to: {filepath}")
            return filepath

        except Exception as e:
            print(f"Error saving result: {str(e)}")
            return None

    def run_comparison(self):
        """Run all extraction methods and compare results"""
        print("Starting Twitter/X Text Extraction Comparison")
        print("="*60)

        # Setup browser
        self.setup_browser()

        try:
            # Load the tweet
            tweet_element = self.load_tweet()
            if not tweet_element:
                print("Failed to load tweet content")
                return

            # Run all methods
            methods = [
                self.method1_inner_text,
                self.method2_javascript_evaluation,
                self.method3_accessibility_tree,
                self.method4_text_content_cleanup
            ]

            results = {}

            for i, method in enumerate(methods, 1):
                print(f"\n{'='*60}")
                print(f"Running Method {i}/{len(methods)}")
                print('='*60)

                result = method(tweet_element)
                method_name = f"Method {i}: {method.__name__.replace('method', '').replace('_', ' ').title()}"
                results[method_name] = result

                # Save individual result
                self.save_result_to_file(method_name, result)

                # Small delay between methods
                time.sleep(1)

            # Print comparison summary
            self.print_comparison_summary(results)

        finally:
            self.cleanup_browser()

    def print_comparison_summary(self, results):
        """Print a summary comparing all methods"""
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)

        successful_methods = []
        failed_methods = []

        for method_name, result in results.items():
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            text_length = len(result['text']) if result['text'] else 0

            print(f"\n{method_name}")
            print(f"  Status: {status}")
            print(f"  Time: {result['time']:.2f}s")
            print(f"  Text Length: {text_length} chars")

            if result['error']:
                print(f"  Error: {result['error']}")
                failed_methods.append(method_name)
            else:
                successful_methods.append(method_name)

                # Show unique words count for successful methods
                if result['text']:
                    words = set(result['text'].lower().split())
                    print(f"  Unique Words: {len(words)}")

        print(f"\n{'='*60}")
        print(f"Successful Methods: {len(successful_methods)}/{len(results)}")
        print(f"Failed Methods: {len(failed_methods)}/{len(results)}")

        if successful_methods:
            print(f"\nBest performing method(s):")
            # Sort by text length (longer is usually more complete)
            successful_methods.sort(key=lambda x: len(results[x]['text']), reverse=True)
            for method in successful_methods[:3]:  # Top 3
                length = len(results[method]['text'])
                print(f"  - {method}: {length} chars")

def main():
    """Main function to run the comparison"""
    comparison = TwitterExtractionComparison()
    comparison.run_comparison()

if __name__ == "__main__":
    main()