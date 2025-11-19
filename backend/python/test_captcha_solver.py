import os
from captcha_solver import CAPTCHARecognizer, CaptchaSolver

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'captcha_model_best.keras')
CONFIG_PATH = os.path.join(MODEL_DIR, 'model_config.json')

print('Testing CAPTCHA solver initialization...')

# Initialize recognizer (graceful if model missing)
rec = CAPTCHARecognizer(MODEL_PATH, CONFIG_PATH)
print('Model loaded:', rec.model_loaded)
if rec.model_loaded:
    print('Model input size: ', rec.img_height, 'x', rec.img_width)

# Test fallback solver
s = CaptchaSolver()
print('Fallback solver created. Using AI recognizer inside fallback:', bool(s.recognizer))
print('Done.')
