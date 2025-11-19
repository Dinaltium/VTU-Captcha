"""
Configuration file for VTU Results Fetcher
Adjust these settings based on your needs
"""

# Captcha Model Configuration
CAPTCHA_CONFIG = {
    'model_path': 'captcha_model_best.h5',
    'char_map_path': 'char_to_int.json',
    
    # Image preprocessing settings
    'image_size': (160, 75),  # (width, height) - adjust based on your model
    'grayscale': True,
    'normalize': True,  # Divide by 255
    
    # Model prediction settings
    'captcha_length': 6,  # VTU captcha is typically 6 characters
    'verbose': False,  # Set to True for debugging
}

# Selenium Configuration
SELENIUM_CONFIG = {
    'headless': False,  # Set to True to hide browser window
    'timeout': 10,  # Wait timeout in seconds
    'max_retries': 3,  # Max captcha solve attempts
    'implicit_wait': 5,  # Implicit wait for elements
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,  # Set to False in production
    'cors_enabled': True,
}

# VTU Website Configuration
VTU_CONFIG = {
    'base_url': 'https://results.vtu.ac.in',
    'timeout_minutes': 5,  # Results page timeout
}

# USN Validation
USN_PATTERN = r'^\d[A-Z]{2}\d{2}[A-Z]{2}\d{3}$'

# College Codes (Add more as needed)
VALID_COLLEGE_CODES = [
    'PA', 'MN', 'JK', 'SE', 'BM', 'AT', 'CS', 'KA', 'HU', 'VV',
    'BT', 'GD', 'MH', 'NK', 'PV', 'RN', 'SJ', 'TN', 'YM', 'AB',
    # Add more college codes from the PDF
]

# Branch Codes
VALID_BRANCH_CODES = [
    'CS', 'IS', 'EC', 'EE', 'ME', 'CV', 'CH', 'BT', 'IM', 'AE',
    'AU', 'EN', 'IE', 'ML', 'IC', 'ET', 'TC', 'IQ', 'IP',
]

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'log_file': 'vtu_fetcher.log',
}
