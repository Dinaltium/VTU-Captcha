"""
VTU Results Fetcher using Selenium with AI CAPTCHA Recognition
Automates the process of fetching results from results.vtu.ac.in
Now with integrated CRNN model for automatic CAPTCHA solving
"""
import time
import re
import warnings
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, UnexpectedAlertPresentException
from captcha_solver import CaptchaSolver
from sgpa_calculator import SGPACalculator
from webdriver_manager.chrome import ChromeDriverManager
import shutil
import stat
import json
import base64
from io import BytesIO
from PIL import Image

# Suppress SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class VTUResultsFetcher:
    def __init__(self, headless=False, captcha_recognizer=None):
        """
        Initialize the VTU Results Fetcher
        
        Args:
            headless: Run browser in headless mode
            captcha_recognizer: Optional AI CAPTCHARecognizer instance for automatic solving
        """
        self.base_url = "https://results.vtu.ac.in"
        self.driver = None
        self._captcha_solver = None  # Lazy initialization (fallback)
        self.headless = headless
        
        # AI CAPTCHA Recognition
        self.captcha_recognizer = captcha_recognizer  # AI model
        self.max_captcha_attempts = 3  # Max attempts for AI recognition
        self.captcha_confidence_threshold = 0.70  # Minimum confidence to accept prediction
        
        # Track CAPTCHA solving statistics
        self.captcha_stats = {
            'total_attempts': 0,
            'ai_success': 0,
            'ai_failed': 0,
            'fallback_used': 0
        }
    
    @property
    def captcha_solver(self):
        """Lazy load captcha solver only when needed (fallback method)"""
        if self._captcha_solver is None:
            self._captcha_solver = CaptchaSolver()
        return self._captcha_solver
        
    def setup_driver(self):
        """Set up Chrome WebDriver with options"""
        from selenium.webdriver.chrome.service import Service
        import os
        
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Use local chromedriver from backend folder (Linux/Windows compatible)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(script_dir)
        
        # Check for chromedriver (Linux) or chromedriver.exe (Windows)
        import platform
        if platform.system() == 'Windows':
            chromedriver_name = 'chromedriver.exe'
        else:
            chromedriver_name = 'chromedriver'
        
        chromedriver_path = os.path.join(backend_dir, chromedriver_name)
        
        # If local chromedriver not found, try system PATH
        if os.path.exists(chromedriver_path):
            service = Service(chromedriver_path)
        else:
            print(f"⚠️  Local chromedriver not found at {chromedriver_path}")
            print("   Downloading matching ChromeDriver via webdriver-manager...")
            try:
                driver_path = ChromeDriverManager().install()
                # Copy into backend for reuse
                try:
                    shutil.copy2(driver_path, chromedriver_path)
                    os.chmod(chromedriver_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR |
                                               stat.S_IRGRP | stat.S_IXGRP |
                                               stat.S_IROTH | stat.S_IXOTH)
                    print(f"✓ Cached ChromeDriver at {chromedriver_path}")
                except Exception as copy_err:
                    print(f"⚠️  Could not cache driver locally: {copy_err}")
                service = Service(driver_path)
                print(f"✓ ChromeDriver downloaded to {driver_path}")
            except Exception as e:
                print(f"✗ webdriver-manager download failed: {e}")
                print("   Falling back to Selenium Manager (may require internet access)")
                service = Service()  # Fall back to selenium-manager managed driver
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.maximize_window()
        
        if self.captcha_recognizer and self.captcha_recognizer.model_loaded:
            print("✓ Chrome WebDriver initialized with AI CAPTCHA recognition")
        else:
            print("✓ Chrome WebDriver initialized (fallback CAPTCHA mode)")

    def _safe_click(self, element, timeout=5):
        """
        Try clicking an element using several fallbacks to avoid
        ElementNotInteractableException: normal click -> ActionChains -> JS click.
        Returns True on success, False otherwise.
        """
        from selenium.webdriver import ActionChains
        try:
            # Wait until element is present and (ideally) clickable
            try:
                WebDriverWait(self.driver, timeout).until(EC.element_to_be_clickable((By.XPATH, self._xpath_of(element))))
            except Exception:
                # If waiting by XPath fails (we don't always have a stable locator), ignore and try clicking anyway
                pass

            # Try normal click first
            try:
                element.click()
                return True
            except Exception:
                pass

            # Scroll into view and try ActionChains click
            try:
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element)
                time.sleep(0.15)
                ActionChains(self.driver).move_to_element(element).click().perform()
                return True
            except Exception:
                pass

            # Final fallback: JavaScript click
            try:
                self.driver.execute_script("arguments[0].click();", element)
                return True
            except Exception:
                return False

        except Exception:
            return False

    def _xpath_of(self, element):
        """
        Helper to derive a simple XPath for an element based on its attributes.
        This is used only for best-effort waiting; it's okay if it fails.
        """
        try:
            tag = element.tag_name
            id_attr = element.get_attribute('id')
            if id_attr:
                return f"//*[{tag}][@id='{id_attr}']"
            cls = element.get_attribute('class')
            if cls:
                # use first class token
                first = cls.split()[0]
                return f"//{tag}[contains(@class, '{first}') ]"
            # fallback to tag
            return f"//{tag}"
        except Exception:
            return "//*"
        
    def get_available_exams(self):
        """Fetch the list of available exams from the main page"""
        try:
            self.driver.get(self.base_url)
            print(f"✓ Navigating to {self.base_url}")
            
            # Wait for the exam list to load
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.ID, "webc")))
            
            # Find all exam panels
            exam_panels = self.driver.find_elements(By.CSS_SELECTOR, ".panel-heading[onclick]")
            
            exams = []
            for idx, panel in enumerate(exam_panels, 1):
                exam_name = panel.text.strip()
                onclick_attr = panel.get_attribute('onclick')
                
                # Extract URL from onclick attribute
                url_match = re.search(r"window\.open\('([^']+)'", onclick_attr)
                if url_match:
                    exam_url = url_match.group(1)
                    exams.append({
                        'id': idx,
                        'name': exam_name,
                        'url': exam_url
                    })
            
            print(f"✓ Found {len(exams)} available exams")
            return exams
            
        except Exception as e:
            print(f"✗ Error fetching exams: {str(e)}")
            return []
    
    def select_exam(self, exam_url):
        """Navigate to the selected exam page"""
        try:
            # Store the current window handle
            main_window = self.driver.current_window_handle
            
            # Navigate to the exam URL
            full_url = f"{self.base_url}/{exam_url}" if not exam_url.startswith('http') else exam_url
            self.driver.get(full_url)
            print(f"✓ Navigating to exam: {full_url}")
            
            time.sleep(2)
            
            # Check if new window/tab was opened
            if len(self.driver.window_handles) > 1:
                # Switch to the new window
                for handle in self.driver.window_handles:
                    if handle != main_window:
                        self.driver.switch_to.window(handle)
                        print("✓ Switched to new tab")
                        break
            
            return True
            
        except Exception as e:
            print(f"✗ Error selecting exam: {str(e)}")
            return False
    
    def get_exam_types_and_branches(self):
        """
        Get exam types (tabs) and branches (sub-exams) from the current page
        Returns: {'types': [...], 'branches': {...}}
        """
        try:
            time.sleep(2)
            result = {'types': [], 'branches': {}}
            
            # Get exam types (tabs like CBCS, CBCS - RV Updated)
            try:
                tabs = self.driver.find_elements(By.CSS_SELECTOR, ".logmod__tabs li")
                for tab in tabs:
                    tab_id = tab.get_attribute('data-tabtar')
                    tab_text = tab.text.strip()
                    if tab_text:
                        result['types'].append({
                            'id': tab_id,
                            'name': tab_text
                        })
                print(f"✓ Found {len(result['types'])} exam types")
            except Exception as e:
                print(f"⚠️  No exam types found: {e}")
            
            # Get branches for each type
            for exam_type in result['types']:
                tab_id = exam_type['id']
                branches = []
                
                try:
                    # Find panels in this tab
                    tab_selector = f".logmod__tab.{tab_id}"
                    panels = self.driver.find_elements(By.CSS_SELECTOR, f"{tab_selector} .panel-heading[onclick]")
                    
                    for i, panel in enumerate(panels):
                        onclick_attr = panel.get_attribute('onclick')
                        text = panel.text.strip()
                        
                        # Clean up text (remove arrow images text)
                        clean_text = re.sub(r'Click here for (updated results of |results of )?', '', text, flags=re.IGNORECASE)
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                        
                        # Use a combination of tab_id, index, and text as unique identifier
                        unique_id = f"{tab_id}:{i}:{clean_text}"
                        
                        branches.append({
                            'name': clean_text,
                            'url': unique_id,
                            'tab_id': tab_id,
                            'index': i,
                            'onclick': onclick_attr
                        })
                    
                    result['branches'][exam_type['id']] = branches
                    print(f"✓ Found {len(branches)} branches for {exam_type['name']}")
                    
                except Exception as e:
                    print(f"⚠️  Error getting branches for {exam_type['id']}: {e}")
                    result['branches'][exam_type['id']] = []
            
            return result
            
        except Exception as e:
            print(f"✗ Error getting exam types and branches: {str(e)}")
            return {'types': [], 'branches': {}}
    
    def check_page_type(self):
        """
        Check if the current page requires another exam selection or USN input
        Returns: 'exam_selection' or 'usn_input' or 'exam_type_selection'
        """
        try:
            time.sleep(2)
            
            # Check for USN input field
            try:
                usn_field = self.driver.find_element(By.NAME, "lns")
                print("✓ Detected USN input page")
                return 'usn_input'
            except NoSuchElementException:
                pass
            
            # Check for exam type tabs (CBCS selection page)
            try:
                tabs = self.driver.find_elements(By.CSS_SELECTOR, ".logmod__tabs li")
                if len(tabs) > 0:
                    print("✓ Detected exam type selection page")
                    return 'exam_type_selection'
            except NoSuchElementException:
                pass
            
            # Check for exam selection panels
            try:
                exam_panels = self.driver.find_elements(By.CSS_SELECTOR, ".panel-heading[onclick]")
                if len(exam_panels) > 0:
                    print("✓ Detected another exam selection page")
                    return 'exam_selection'
            except NoSuchElementException:
                pass
            
            return 'unknown'
            
        except Exception as e:
            print(f"✗ Error checking page type: {str(e)}")
            return 'unknown'
    
    def select_branch(self, branch_unique_id):
        """Navigate to a specific branch by clicking its panel using the unique ID"""
        try:
            print(f"🔍 Looking for branch with unique ID: {branch_unique_id}")
            
            # Parse the unique ID: "tab_id:index:name"
            parts = branch_unique_id.split(':', 2)
            if len(parts) != 3:
                print(f"✗ Invalid branch ID format: {branch_unique_id}")
                return False
            
            tab_id, index_str, branch_name = parts
            target_index = int(index_str)
            
            print(f"   Tab: {tab_id}, Index: {target_index}, Name: {branch_name}")
            
            # Store current window
            main_window = self.driver.current_window_handle
            
            # Find panels in the specific tab
            tab_selector = f".logmod__tab.{tab_id}"
            panels = self.driver.find_elements(By.CSS_SELECTOR, f"{tab_selector} .panel-heading[onclick]")
            print(f"🔍 Found {len(panels)} panels in tab {tab_id}")
            
            if target_index < len(panels):
                panel = panels[target_index]
                panel_text = panel.text.strip()
                onclick_attr = panel.get_attribute('onclick')
                
                print(f"✓ Found target panel at index {target_index}: '{panel_text}'")
                
                before_url = self.driver.current_url
                onclick_attr = onclick_attr or panel.get_attribute('onclick')

                clicked = self._safe_click(panel)
                time.sleep(1)

                # If click didn't change page or open new window, try executing onclick JS as a fallback
                navigated = (self.driver.current_url != before_url) or (len(self.driver.window_handles) > 1)
                if not navigated and onclick_attr:
                    print(f"⚠️  Click didn't navigate; executing onclick JS: {onclick_attr}")
                    try:
                        self.driver.execute_script(onclick_attr)
                    except Exception as e:
                        print(f"⚠️  Executing onclick JS failed: {e}")
                    time.sleep(1)

                # After click or JS, attempt window switch if a new window opened
                if len(self.driver.window_handles) > 1:
                    for handle in self.driver.window_handles:
                        if handle != main_window:
                            self.driver.switch_to.window(handle)
                            print("✓ Switched to new window")
                            break
                else:
                    # If still same URL, log and treat as failure
                    if self.driver.current_url == before_url:
                        print("✗ Click/onclick did not navigate to a new page")
                        return False
                print(f"✓ Clicked branch panel")
                
                return True
            else:
                print(f"✗ Index {target_index} out of range (found {len(panels)} panels)")
                return False
            
        except Exception as e:
            import traceback
            print(f"✗ Error selecting branch: {str(e)}")
            print(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def select_first_branch_in_type(self, exam_type):
        """Navigate to the first branch in a specific exam type (e.g., CBCS or CBCS-RV)"""
        try:
            print(f"🔍 Looking for first branch in type: {exam_type}")
            
            # Store current window
            main_window = self.driver.current_window_handle
            
            # Find the first panel in the specific tab
            tab_selector = f".logmod__tab.{exam_type}"
            panels = self.driver.find_elements(By.CSS_SELECTOR, f"{tab_selector} .panel-heading[onclick]")
            
            if len(panels) == 0:
                print(f"✗ No branches found in type: {exam_type}")
                return False
            
            # Click the first panel
            panel = panels[0]
            panel_text = panel.text.strip()

            print(f"✓ Found first branch: '{panel_text}'")

            before_url = self.driver.current_url
            onclick_attr = panel.get_attribute('onclick')

            clicked = self._safe_click(panel)
            time.sleep(1)

            navigated = (self.driver.current_url != before_url) or (len(self.driver.window_handles) > 1)
            if not navigated and onclick_attr:
                print(f"⚠️  Click didn't navigate; executing onclick JS: {onclick_attr}")
                try:
                    self.driver.execute_script(onclick_attr)
                except Exception as e:
                    print(f"⚠️  Executing onclick JS failed: {e}")
                time.sleep(1)

            if len(self.driver.window_handles) > 1:
                for handle in self.driver.window_handles:
                    if handle != main_window:
                        self.driver.switch_to.window(handle)
                        print("✓ Switched to new window")
                        break
            else:
                if self.driver.current_url == before_url:
                    print("✗ Click/onclick did not navigate to a new page")
                    return False
            print(f"✓ Clicked branch panel")
            
            return True
            
        except Exception as e:
            import traceback
            print(f"✗ Error selecting first branch: {str(e)}")
            print(f"   Traceback: {traceback.format_exc()}")
            return False
    
    # ========================================================================
    # AI CAPTCHA RECOGNITION METHODS
    # ========================================================================
    
    def get_captcha_image_base64(self):
        """
        Get CAPTCHA image as base64 string for AI recognition
        
        Returns:
            str: Base64 encoded image or None if failed
        """
        try:
            # Find captcha image
            captcha_img = self.driver.find_element(By.CSS_SELECTOR, "img[src*='captcha']")
            
            # Method 1: Try screenshot (most reliable)
            try:
                captcha_base64 = captcha_img.screenshot_as_base64
                if captcha_base64 and len(captcha_base64) > 100:
                    return captcha_base64
            except Exception as e:
                print(f"⚠️  Screenshot method failed: {str(e)}")
            
            # Method 2: Try JavaScript canvas
            try:
                captcha_base64 = self.driver.execute_script("""
                    var img = arguments[0];
                    var canvas = document.createElement('canvas');
                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;
                    var ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    return canvas.toDataURL('image/png').split(',')[1];
                """, captcha_img)
                
                if captcha_base64 and len(captcha_base64) > 100:
                    return captcha_base64
            except Exception as e:
                print(f"⚠️  Canvas method failed: {str(e)}")
            
            # Method 3: HTTP download
            try:
                captcha_src = captcha_img.get_attribute('src')
                
                if captcha_src.startswith('/'):
                    current_url = self.driver.current_url
                    from urllib.parse import urlparse
                    parsed = urlparse(current_url)
                    captcha_url = f"{parsed.scheme}://{parsed.netloc}{captcha_src}"
                else:
                    captcha_url = captcha_src
                
                import requests
                session = requests.Session()
                session.verify = False
                
                cookies = {cookie['name']: cookie['value'] for cookie in self.driver.get_cookies()}
                headers = {
                    'User-Agent': self.driver.execute_script("return navigator.userAgent;"),
                    'Referer': self.driver.current_url
                }
                
                response = session.get(captcha_url, cookies=cookies, headers=headers, verify=False)
                response.raise_for_status()
                
                captcha_base64 = base64.b64encode(response.content).decode()
                if captcha_base64 and len(captcha_base64) > 100:
                    return captcha_base64
            except Exception as e:
                print(f"⚠️  HTTP method failed: {str(e)}")
            
            return None
            
        except Exception as e:
            print(f"✗ Error getting CAPTCHA image: {str(e)}")
            return None
    
    def _solve_captcha_with_ai(self):
        """
        Solve CAPTCHA using AI recognition with retry logic
        
        Returns:
            dict with success status, captcha_text, and confidence
        """
        print("\n" + "="*70)
        print("🤖 AI CAPTCHA RECOGNITION")
        print("="*70)
        
        for attempt in range(1, self.max_captcha_attempts + 1):
            try:
                print(f"\n🔄 Attempt {attempt}/{self.max_captcha_attempts}")
                self.captcha_stats['total_attempts'] += 1
                
                # Wait for CAPTCHA to load
                time.sleep(1)
                
                # Get CAPTCHA image as base64
                captcha_base64 = self.get_captcha_image_base64()
                
                if not captcha_base64:
                    print("✗ Failed to capture CAPTCHA image")
                    continue
                
                # Recognize using AI model
                print("🤖 Running AI recognition...")
                recognition_result = self.captcha_recognizer.recognize_from_base64(captcha_base64)
                
                if not recognition_result['success']:
                    print(f"✗ AI recognition failed: {recognition_result.get('error', 'Unknown error')}")
                    continue
                
                captcha_text = recognition_result['text']
                confidence = recognition_result['confidence']
                
                print(f"📝 Predicted: '{captcha_text}'")
                print(f"📊 Confidence: {confidence*100:.2f}%")
                
                # Check confidence threshold
                if confidence < self.captcha_confidence_threshold:
                    print(f"⚠️  Confidence below threshold ({self.captcha_confidence_threshold*100:.0f}%)")
                    
                    # Refresh CAPTCHA for next attempt
                    if attempt < self.max_captcha_attempts:
                        print("🔄 Refreshing CAPTCHA...")
                        self._refresh_captcha()
                        time.sleep(1)
                    continue
                
                # Enter CAPTCHA
                captcha_input = self.driver.find_element(By.NAME, "captchacode")
                captcha_input.clear()
                captcha_input.send_keys(captcha_text)
                print(f"✓ Entered CAPTCHA: '{captcha_text}'")
                
                # Submit form
                submit_button = self.driver.find_element(By.ID, "submit")
                submit_button.click()
                print("✓ Form submitted")
                
                # Wait and check for errors
                time.sleep(3)
                
                # Check for alert or error
                if self._check_for_captcha_error():
                    print("✗ CAPTCHA was incorrect")
                    self.captcha_stats['ai_failed'] += 1
                    
                    if attempt < self.max_captcha_attempts:
                        print("🔄 Retrying...")
                        time.sleep(1)
                    continue
                else:
                    # Success!
                    self.captcha_stats['ai_success'] += 1
                    print("✓ CAPTCHA solved successfully!")
                    print("="*70 + "\n")
                    return {
                        'success': True,
                        'captcha_text': captcha_text,
                        'captcha_confidence': confidence,
                        'attempts': attempt,
                        'method': 'ai'
                    }
                    
            except Exception as e:
                print(f"✗ Error in attempt {attempt}: {e}")
                if attempt < self.max_captcha_attempts:
                    time.sleep(1)
                continue
        
        # All AI attempts failed
        print("✗ AI CAPTCHA solving failed after all attempts")
        print("="*70 + "\n")
        return {
            'success': False,
            'error': 'All AI CAPTCHA attempts failed',
            'attempts': self.max_captcha_attempts,
            'method': 'ai'
        }
    
    def _refresh_captcha(self):
        """Refresh CAPTCHA image"""
        try:
            # Look for refresh button/link
            refresh_selectors = [
                "a[onclick*='captcha']",
                "button[onclick*='captcha']",
                "img[onclick*='captcha']",
                "#captchaRefresh",
                ".captcha-refresh"
            ]
            
            for selector in refresh_selectors:
                try:
                    refresh_elem = self.driver.find_element(By.CSS_SELECTOR, selector)
                    refresh_elem.click()
                    time.sleep(1)
                    return
                except:
                    continue
            
            # If no refresh button found, try reloading page
            print("⚠️  No refresh button found, reloading page...")
            self.driver.refresh()
            time.sleep(2)
            
            # Re-enter USN after refresh
            try:
                usn_field = self.driver.find_element(By.NAME, "lns")
                current_usn = usn_field.get_attribute('value')
                if not current_usn:
                    # USN was cleared, but we don't have it stored here
                    # This should be handled in the calling function
                    pass
            except:
                pass
                
        except Exception as e:
            print(f"⚠️  Error refreshing CAPTCHA: {e}")
    
    def _check_for_captcha_error(self):
        """
        Check if CAPTCHA submission resulted in an error
        
        Returns:
            bool: True if error detected, False otherwise
        """
        try:
            # Check for JavaScript alert
            try:
                alert = self.driver.switch_to.alert
                alert_text = alert.text.lower()
                alert.accept()
                
                # Check if it's a CAPTCHA error (not USN error)
                captcha_error_keywords = ['captcha', 'invalid code', 'wrong code']
                if any(keyword in alert_text for keyword in captcha_error_keywords):
                    return True
                
                # If it's an USN error, that's a different issue
                if 'not available' in alert_text or 'invalid usn' in alert_text:
                    return False  # This will be handled elsewhere
                    
            except:
                pass  # No alert
            
            # Check if we're still on the form page
            try:
                self.driver.find_element(By.NAME, "captchacode")
                # CAPTCHA field still present = still on form = likely error
                return True
            except:
                # CAPTCHA field gone = moved to results page = success
                return False
                
        except Exception as e:
            print(f"⚠️  Error checking for CAPTCHA error: {e}")
            return False
    
    # ========================================================================
    # MAIN CAPTCHA SUBMISSION METHOD (WITH AI INTEGRATION)
    # ========================================================================
    
    def download_captcha(self):
        """Download the captcha image and return bytes (fallback method)"""
        try:
            # Find captcha image
            captcha_img = self.driver.find_element(By.CSS_SELECTOR, "img[src*='captcha']")
            captcha_src = captcha_img.get_attribute('src')
            
            # Method 1: Screenshot
            try:
                captcha_base64_str = captcha_img.screenshot_as_base64
                captcha_bytes = base64.b64decode(captcha_base64_str)
                if captcha_bytes and len(captcha_bytes) > 0:
                    return captcha_bytes
            except Exception as e:
                print(f"⚠️  Screenshot method failed: {str(e)}")
            
            # Method 2: Canvas
            try:
                captcha_base64_str = self.driver.execute_script("""
                    var img = arguments[0];
                    var canvas = document.createElement('canvas');
                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;
                    var ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    return canvas.toDataURL('image/png').split(',')[1];
                """, captcha_img)
                
                if captcha_base64_str:
                    captcha_bytes = base64.b64decode(captcha_base64_str)
                    if captcha_bytes and len(captcha_bytes) > 0:
                        return captcha_bytes
            except Exception as e:
                print(f"⚠️  Canvas method failed: {str(e)}")
            
            # Method 3: HTTP request
            try:
                if captcha_src.startswith('/'):
                    current_url = self.driver.current_url
                    from urllib.parse import urlparse
                    parsed = urlparse(current_url)
                    captcha_url = f"{parsed.scheme}://{parsed.netloc}{captcha_src}"
                else:
                    captcha_url = captcha_src
                
                import requests
                session = requests.Session()
                session.verify = False
                
                cookies = {cookie['name']: cookie['value'] for cookie in self.driver.get_cookies()}
                headers = {
                    'User-Agent': self.driver.execute_script("return navigator.userAgent;"),
                    'Referer': self.driver.current_url
                }
                
                response = session.get(captcha_url, cookies=cookies, headers=headers, verify=False)
                response.raise_for_status()
                
                captcha_bytes = response.content
                if captcha_bytes and len(captcha_bytes) > 0:
                    return captcha_bytes
            except Exception as e:
                print(f"⚠️  HTTP method failed: {str(e)}")
            
            return None
            
        except Exception as e:
            import traceback
            print(f"✗ Error downloading captcha: {str(e)}")
            return None
    
    def submit_usn_and_captcha(self, usn, max_retries=3):
        """
        Submit USN and solve captcha to fetch results
        Now with AI CAPTCHA recognition support
        
        Args:
            usn: University Seat Number
            max_retries: Maximum retry attempts (used only for fallback method)
            
        Returns:
            dict with success/error status and captcha info
        """
        try:
            # Wait for page to load
            wait = WebDriverWait(self.driver, 10)
            
            # Find and fill USN field
            usn_field = wait.until(EC.presence_of_element_located((By.NAME, "lns")))
            usn_field.clear()
            usn_field.send_keys(usn)
            print(f"✓ USN entered: {usn}")
            
            # Try AI CAPTCHA recognition if available
            if self.captcha_recognizer and self.captcha_recognizer.model_loaded:
                result = self._solve_captcha_with_ai()
                
                if result['success']:
                    return result
                else:
                    # AI failed, try fallback if available
                    print(f"⚠️  AI CAPTCHA failed: {result.get('error', 'Unknown error')}")
                    
                    # Fall back to traditional method
                    print("⚠️  Attempting fallback CAPTCHA solving...")
                    self.captcha_stats['fallback_used'] += 1
                    return self._submit_with_fallback_captcha(usn, max_retries)
            else:
                # No AI available, use fallback method
                print("⚠️  No AI CAPTCHA recognizer - using fallback method")
                self.captcha_stats['fallback_used'] += 1
                return self._submit_with_fallback_captcha(usn, max_retries)
                
        except Exception as e:
            return {
                'success': False,
                'error': 'submission_failed',
                'message': str(e)
            }
    
    def _submit_with_fallback_captcha(self, usn, max_retries=3):
        """Fallback CAPTCHA solving using traditional CaptchaSolver"""
        for attempt in range(max_retries):
            try:
                print(f"\n{'='*50}")
                print(f"Fallback Attempt {attempt + 1} of {max_retries}")
                print(f"{'='*50}")
                
                # Re-enter USN (may have been cleared)
                try:
                    usn_field = self.driver.find_element(By.NAME, "lns")
                    if not usn_field.get_attribute('value'):
                        usn_field.clear()
                        usn_field.send_keys(usn)
                except:
                    pass
                
                # Download and solve captcha
                captcha_bytes = self.download_captcha()
                
                if not captcha_bytes:
                    print(f"✗ Failed to download captcha")
                    continue
                
                captcha_text = self.captcha_solver.solve(captcha_bytes)
                
                if not captcha_text:
                    print("✗ Failed to solve captcha")
                    continue
                
                # Fill captcha field
                captcha_field = self.driver.find_element(By.NAME, "captchacode")
                captcha_field.clear()
                captcha_field.send_keys(captcha_text)
                print(f"✓ Captcha entered: {captcha_text}")
                
                # Click submit button
                submit_btn = self.driver.find_element(By.ID, "submit")
                submit_btn.click()
                print("✓ Form submitted")
                
                time.sleep(3)
                
                # Check for alerts (invalid USN or wrong captcha)
                try:
                    alert = self.driver.switch_to.alert
                    alert_text = alert.text
                    alert.accept()
                    
                    if "not available" in alert_text.lower() or "invalid" in alert_text.lower():
                        return {'error': 'invalid_usn', 'message': alert_text}
                    else:
                        print(f"⚠️  Alert: {alert_text}")
                        continue
                        
                except:
                    # No alert means success
                    print("✓ Form submitted successfully (fallback method)")
                    return {
                        'success': True,
                        'method': 'fallback',
                        'attempts': attempt + 1
                    }
                
            except UnexpectedAlertPresentException as e:
                try:
                    alert = self.driver.switch_to.alert
                    alert_text = alert.text
                    alert.accept()
                    print(f"⚠️  Alert: {alert_text}")
                    
                    if "not available" in alert_text.lower() or "invalid" in alert_text.lower():
                        return {'error': 'invalid_usn', 'message': alert_text}
                except:
                    pass
                    
            except Exception as e:
                print(f"✗ Error in fallback attempt {attempt + 1}: {str(e)}")
                
        return {
            'error': 'max_retries',
            'message': 'Failed to submit form after maximum retries (fallback method)'
        }
    
    # ========================================================================
    # RESULTS EXTRACTION
    # ========================================================================
    
    def extract_results(self, fetch_details=True):
        """Extract results from the results page"""
        try:
            wait = WebDriverWait(self.driver, 10)
            
            # Wait for results table to load
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "divTable")))
            print("✓ Results page loaded")
            
            # Extract student info
            student_info = {}
            try:
                page_text = self.driver.page_source
                student_info['usn'] = re.search(r'USN\s*:\s*(\w+)', page_text, re.IGNORECASE)
                student_info['name'] = re.search(r'Name\s*:\s*([^<\n]+)', page_text, re.IGNORECASE)
                
                if student_info['usn']:
                    student_info['usn'] = student_info['usn'].group(1).strip()
                if student_info['name']:
                    student_info['name'] = student_info['name'].group(1).strip()
            except:
                pass
            
            # Extract semester info
            semester = None
            try:
                sem_div = self.driver.find_element(By.XPATH, "//div[contains(text(), 'Semester :')]")
                semester = re.search(r'Semester\s*:\s*(\d+)', sem_div.text)
                if semester:
                    semester = int(semester.group(1))
            except:
                pass
            
            # Extract subject results
            subjects = []
            subject_rows = self.driver.find_elements(By.CSS_SELECTOR, ".divTableRow")[1:]  # Skip header
            
            for row in subject_rows:
                cells = row.find_elements(By.CLASS_NAME, "divTableCell")
                if len(cells) >= 6:
                    subject = {
                        'code': cells[0].text.strip(),
                        'name': cells[1].text.strip(),
                        'internal': int(cells[2].text.strip()) if cells[2].text.strip().isdigit() else 0,
                        'external': int(cells[3].text.strip()) if cells[3].text.strip().isdigit() else 0,
                        'total': int(cells[4].text.strip()) if cells[4].text.strip().isdigit() else 0,
                        'result': cells[5].text.strip(),
                        'date': cells[6].text.strip() if len(cells) > 6 else ''
                    }
                    subjects.append(subject)
            
            print(f"✓ Extracted {len(subjects)} subjects")
            
            # Prepare results data
            results = {
                'student_info': student_info,
                'semester': semester,
                'subjects': subjects
            }
            
            if fetch_details:
                # Calculate SGPA and statistics
                subjects_for_sgpa = [{'total_marks': s['total']} for s in subjects]
                sgpa_data = SGPACalculator.calculate_sgpa(subjects_for_sgpa)
                
                # Count passed and failed subjects
                passed = [s for s in subjects if s['result'].upper() == 'P']
                failed = [s for s in subjects if s['result'].upper() in ['F', 'A']]
                
                # Add colors to subjects
                for subject in subjects:
                    subject['color'] = SGPACalculator.get_marks_color(subject['total'])
                    subject['grade'], subject['grade_point'] = SGPACalculator.marks_to_grade(subject['total'])
                
                # Performance message
                failed_subjects = [{'name': s['name'], 'code': s['code']} for s in failed]
                performance_msg = SGPACalculator.get_performance_message(
                    sgpa_data['sgpa'], failed_subjects
                )
                
                results['analysis'] = {
                    'sgpa': sgpa_data['sgpa'],
                    'total_marks': sgpa_data['total_marks'],
                    'total_subjects': len(subjects),
                    'passed_subjects': len(passed),
                    'failed_subjects': len(failed),
                    'failed_subject_names': failed_subjects,
                    'performance_message': performance_msg
                }
                
                print(f"\n{'='*50}")
                print(f"SGPA: {sgpa_data['sgpa']}")
                print(f"Total Marks: {sgpa_data['total_marks']}")
                print(f"Passed: {len(passed)} | Failed: {len(failed)}")
                print(f"{performance_msg}")
                print(f"{'='*50}\n")
            
            # Add CAPTCHA statistics if AI was used
            if self.captcha_stats['total_attempts'] > 0:
                results['captcha_stats'] = self.captcha_stats.copy()
            
            return results
            
        except Exception as e:
            print(f"✗ Error extracting results: {str(e)}")
            return None
    
    def download_result(self):
        """Click print button to download/print the result"""
        try:
            print_btn = self.driver.find_element(By.CSS_SELECTOR, "input[onclick='window.print();']")
            print_btn.click()
            print("✓ Print dialog opened")
            time.sleep(2)
            return True
        except Exception as e:
            print(f"✗ Error opening print dialog: {str(e)}")
            return False
    
    def check_timeout(self):
        """Check if the page has timed out and redirected"""
        try:
            current_url = self.driver.current_url
            if current_url == self.base_url or 'index' in current_url.lower():
                return True
            return False
        except:
            return False
    
    def get_captcha_statistics(self):
        """Get CAPTCHA solving statistics"""
        stats = self.captcha_stats.copy()
        
        if stats['total_attempts'] > 0:
            stats['ai_success_rate'] = (stats['ai_success'] / stats['total_attempts']) * 100
        else:
            stats['ai_success_rate'] = 0.0
        
        return stats
    
    def print_captcha_statistics(self):
        """Print CAPTCHA solving statistics"""
        stats = self.get_captcha_statistics()
        
        if stats['total_attempts'] == 0:
            print("\nNo CAPTCHA attempts recorded")
            return
        
        print("\n" + "="*70)
        print("CAPTCHA SOLVING STATISTICS")
        print("="*70)
        print(f"Total Attempts:     {stats['total_attempts']}")
        print(f"AI Success:         {stats['ai_success']}")
        print(f"AI Failed:          {stats['ai_failed']}")
        print(f"Fallback Used:      {stats['fallback_used']}")
        print(f"AI Success Rate:    {stats['ai_success_rate']:.1f}%")
        print("="*70 + "\n")
    
    def close(self):
        """Close the browser"""
        if self.driver:
            # Print statistics before closing
            if self.captcha_stats['total_attempts'] > 0:
                self.print_captcha_statistics()
            
            self.driver.quit()
            print("✓ Browser closed")


def main():
    """Main function for testing"""
    fetcher = VTUResultsFetcher(headless=False)
    
    try:
        # Setup driver
        fetcher.setup_driver()
        
        # Get available exams
        exams = fetcher.get_available_exams()
        
        if not exams:
            print("No exams found!")
            return
        
        # Display exams
        print("\n" + "="*50)
        print("AVAILABLE EXAMS:")
        print("="*50)
        for exam in exams:
            print(f"{exam['id']}. {exam['name']}")
        print("="*50 + "\n")
        
        # User selects exam
        exam_choice = int(input("Select exam number: "))
        selected_exam = next((e for e in exams if e['id'] == exam_choice), None)
        
        if not selected_exam:
            print("Invalid exam selection!")
            return
        
        # Navigate to exam
        fetcher.select_exam(selected_exam['url'])
        
        # Check page type
        page_type = fetcher.check_page_type()
        
        if page_type == 'exam_type_selection':
            # Get types and branches
            types_and_branches = fetcher.get_exam_types_and_branches()
            
            print("\n" + "="*50)
            print("SELECT EXAM TYPE:")
            print("="*50)
            for i, exam_type in enumerate(types_and_branches['types'], 1):
                print(f"{i}. {exam_type['name']}")
            print("="*50 + "\n")
            
            type_choice = int(input("Select exam type: "))
            selected_type = types_and_branches['types'][type_choice - 1]
            
            # Show branches for selected type
            branches = types_and_branches['branches'][selected_type['id']]
            
            print("\n" + "="*50)
            print(f"BRANCHES FOR {selected_type['name']}:")
            print("="*50)
            for i, branch in enumerate(branches, 1):
                print(f"{i}. {branch['name']}")
            print("="*50 + "\n")
            
            branch_choice = int(input("Select branch: "))
            selected_branch = branches[branch_choice - 1]
            
            # Navigate to branch
            fetcher.select_branch(selected_branch['url'])
        
        elif page_type == 'exam_selection':
            # Another exam selection required
            sub_exams = fetcher.get_available_exams()
            print("\n" + "="*50)
            print("SELECT EXAM TYPE:")
            print("="*50)
            for exam in sub_exams:
                print(f"{exam['id']}. {exam['name']}")
            print("="*50 + "\n")
            
            sub_exam_choice = int(input("Select exam type: "))
            selected_sub_exam = next((e for e in sub_exams if e['id'] == sub_exam_choice), None)
            
            if selected_sub_exam:
                fetcher.select_exam(selected_sub_exam['url'])
        
        # Get USN from user
        usn = input("\nEnter USN (e.g., 4PA23CS102): ").strip().upper()
        
        # Validate USN format
        usn_pattern = r'^\d[A-Z]{2}\d{2}[A-Z]{2}\d{3}'
        if not re.match(usn_pattern, usn):
            print("✗ Invalid USN format! Expected format: #@@##@@###")
            return
        
        # Submit form (will use AI CAPTCHA if available)
        result = fetcher.submit_usn_and_captcha(usn)
        
        if 'error' in result:
            print(f"\n✗ Error: {result['message']}")
            return
        
        # Show CAPTCHA method used
        if result.get('method') == 'ai':
            print(f"\n✓ Used AI CAPTCHA recognition (confidence: {result.get('captcha_confidence', 0)*100:.1f}%)")
        elif result.get('method') == 'fallback':
            print("\n✓ Used fallback CAPTCHA method")
        
        # Extract results
        print("\n" + "="*50)
        print("FETCHING RESULTS...")
        print("="*50 + "\n")
        
        fetch_details = input("Fetch detailed analysis? (y/n): ").lower() == 'y'
        results = fetcher.extract_results(fetch_details=fetch_details)
        
        if results:
            print("\n✓ Results fetched successfully!")
            print(json.dumps(results, indent=2))
            
            # Ask if user wants to download
            download = input("\nDownload result? (y/n): ").lower() == 'y'
            if download:
                fetcher.download_result()
        
        # Keep browser open for a while
        input("\nPress Enter to close browser...")
        
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()