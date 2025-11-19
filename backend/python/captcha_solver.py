"""
Flask Backend API for VTU Results Fetcher with AI CAPTCHA Recognition
Integrates the trained CRNN model for automatic CAPTCHA solving
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import json
import numpy as np
from PIL import Image
import io
import base64

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import threading
import re
import time

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app)

# Store active sessions
active_sessions = {}


# ============================================================================
# Custom layers / functions used by the trained model
# ============================================================================

def conv_to_rnn(inp):
    """
    Reshape conv feature maps to sequence format expected by the CRNN's RNN block.
    Matches the function used during training (see train/app.py).
    """
    shape = tf.shape(inp)
    batch, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    x = tf.transpose(inp, perm=[0, 2, 1, 3])  # (batch, width, height, channels)
    x = tf.reshape(x, [batch, width, height * channels])  # (batch, width, height*channels)
    return x

# ============================================================================
# CAPTCHA MODEL CONFIGURATION
# ============================================================================

class CAPTCHARecognizer:
    """AI-powered CAPTCHA recognition using trained CRNN model"""
    
    def __init__(self, model_path, config_path):
        """
        Initialize CAPTCHA recognizer
        
        Args:
            model_path: Path to trained Keras model (.keras file)
            config_path: Path to model configuration JSON
        """
        self.model = None
        self.config = None
        self.char_to_int = None
        self.int_to_char = None
        self.model_loaded = False
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Load character mappings
            config_dir = os.path.dirname(config_path)
            with open(os.path.join(config_dir, 'char_to_int.json'), 'r') as f:
                self.char_to_int = json.load(f)
            with open(os.path.join(config_dir, 'int_to_char.json'), 'r') as f:
                # Convert string keys to int
                int_to_char_raw = json.load(f)
                self.int_to_char = {int(k): v for k, v in int_to_char_raw.items()}
            
            # Load model
            self.model = keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={"custom>conv_to_rnn": conv_to_rnn}
            )
            
            # Extract config values
            self.img_height = self.config['img_height']
            self.img_width = self.config['img_width']
            self.captcha_length = self.config['captcha_length']
            self.characters = self.config['characters']
            self.num_classes = self.config['num_classes']
            
            self.model_loaded = True
            print("✓ CAPTCHA Recognition Model loaded successfully")
            print(f"  - Image size: {self.img_height}×{self.img_width}")
            print(f"  - Expected length: {self.captcha_length}")
            print(f"  - Character set size: {len(self.characters)}")
            print(f"  - Model accuracy: {self.config.get('char_accuracy', 0)*100:.2f}%")
            
        except Exception as e:
            print(f"⚠️  Failed to load CAPTCHA model: {e}")
            print("   CAPTCHA recognition will not be available")
            self.model_loaded = False
    
    def preprocess_image(self, image_data):
        """
        Preprocess image for model input - MUST match training preprocessing exactly
        
        Args:
            image_data: PIL Image or numpy array or bytes
            
        Returns:
            numpy array of shape (1, height, width, 1) normalized to [0, 1]
        """
        # Convert to PIL Image if needed
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        else:
            image = image_data
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to model input size (width, height) for PIL
        image = image.resize((self.img_width, self.img_height), Image.Resampling.BILINEAR)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # CRITICAL: Normalize to [0, 1] range to match training preprocessing
        # Training uses: img = tf.cast(img, tf.float32) / 255.0
        img_array = img_array / 255.0
        
        # Add batch and channel dimensions: (height, width) -> (1, height, width, 1)
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def decode_predictions(self, predictions):
        """
        Decode CTC predictions to text - Keras 3.x compatible
        
        Args:
            predictions: Model output logits of shape (batch, timesteps, num_classes)
            
        Returns:
            List of decoded text strings
        """
        # Apply softmax to get probabilities
        probs = tf.nn.softmax(predictions, axis=-1).numpy()

        # Get input length for CTC decoder (all sequences have same length)
        input_len = np.ones(probs.shape[0], dtype=np.int32) * probs.shape[1]
        
        # Get blank index (last class)
        blank_idx = self.num_classes - 1

        # CTC greedy decode - Keras 3.x compatible
        try:
            decoded_result = keras.backend.ctc_decode(
                probs,
                input_length=input_len,
                greedy=True
            )
            # Keras 3.x returns tuple: (decoded, log_probabilities)
            if isinstance(decoded_result, tuple):
                decoded = decoded_result[0][0].numpy()
            else:
                decoded = decoded_result[0].numpy()
        except Exception as e:
            print(f"⚠️  CTC decode error: {e}, using fallback method")
            # Fallback: argmax + collapse repeats + remove blank
            decoded = []
            for b in range(probs.shape[0]):
                seq = np.argmax(probs[b], axis=-1)
                # Collapse repeats and remove blank
                collapsed = []
                prev = None
                for idx in seq:
                    if idx == prev:
                        continue
                    prev = idx
                    if idx != blank_idx and 0 <= idx < len(self.characters):
                        collapsed.append(idx)
                decoded.append(collapsed)
            decoded = np.array(decoded, dtype=object)

        # Convert indices to characters
        texts = []
        for row in decoded:
            text = ""
            # Handle both dense arrays and ragged arrays
            if isinstance(row, np.ndarray):
                for idx in row:
                    idx = int(idx)
                    if idx >= 0 and idx < len(self.characters):
                        text += self.characters[idx]
            else:
                # Handle list or other iterable
                for idx in row:
                    idx = int(idx)
                    if idx >= 0 and idx < len(self.characters):
                        text += self.characters[idx]
            # Truncate to expected length
            texts.append(text[:self.captcha_length])

        return texts

    def _argmax_collapse(self, predictions):
        """
        Simple argmax + repeat-collapse + blank removal fallback.
        Returns: decoded string
        """
        # predictions: numpy array (1, T, C) or tf.Tensor
        if isinstance(predictions, tf.Tensor):
            probs = tf.nn.softmax(predictions, axis=-1).numpy()
        else:
            probs = tf.nn.softmax(predictions, axis=-1).numpy() if len(predictions.shape) == 3 else predictions

        seq = np.argmax(probs[0], axis=-1)
        blank_idx = self.num_classes - 1

        collapsed = []
        last = None
        for i in seq:
            if i == last:
                continue
            if i == blank_idx:
                last = i
                continue
            if 0 <= i < len(self.characters):
                collapsed.append(self.characters[int(i)])
            last = i

        return ''.join(collapsed)[:self.captcha_length]

    def _beam_decode(self, predictions, beam_width=5):
        """
        Use TensorFlow's CTC beam search decoder as a fallback.
        Returns decoded string for first batch element or empty string on failure.
        """
        try:
            # Convert to logits if needed
            if isinstance(predictions, tf.Tensor):
                probs = tf.nn.softmax(predictions, axis=-1)
            else:
                probs = tf.nn.softmax(predictions, axis=-1)
            
            # predictions shape: (batch, T, C)
            logits = tf.math.log(tf.clip_by_value(probs, 1e-9, 1.0))
            # time-major for ctc: (T, batch, C)
            inputs = tf.transpose(logits, [1, 0, 2])
            sequence_length = tf.fill((probs.shape[0],), probs.shape[1])
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                inputs, 
                sequence_length=sequence_length, 
                beam_width=beam_width, 
                top_paths=1
            )
            dense_decoded = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
            row = dense_decoded[0]
            text = ''
            blank_idx = self.num_classes - 1
            for idx in row:
                idx_int = int(idx)
                if idx_int >= 0 and idx_int < len(self.characters) and idx_int != blank_idx:
                    text += self.characters[idx_int]
            return text[:self.captcha_length]
        except Exception as e:
            print(f"⚠️  Beam decode error: {e}")
            return ''
    
    def recognize(self, image_data):
        """
        Recognize CAPTCHA text from image
        
        Args:
            image_data: Image data (PIL Image, numpy array, or bytes)
            
        Returns:
            dict with 'success', 'text', 'confidence' keys
        """
        if not self.model_loaded:
            return {
                'success': False,
                'error': 'Model not loaded',
                'text': None,
                'confidence': 0.0
            }
        
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_data)
            
            # Get predictions
            predictions = self.model.predict(img_array, verbose=0)

            # Primary decode (greedy)
            texts = self.decode_predictions(predictions)

            predicted_text = texts[0] if texts else ''

            # Calculate confidence (average max probability per timestep)
            probs = tf.nn.softmax(predictions[0], axis=-1).numpy()
            confidence = float(np.mean(np.max(probs, axis=-1)))

            # If length is wrong, try fallbacks
            if len(predicted_text) != self.captcha_length:
                print(f"⚠️  Greedy decode length {len(predicted_text)} != expected {self.captcha_length}. Trying beam/argmax fallbacks...")

                # Try beam search
                beam_text = self._beam_decode(predictions, beam_width=8)
                if beam_text and len(beam_text) == self.captcha_length:
                    print(f"✓ Beam decode success: {beam_text}")
                    return {'success': True, 'text': beam_text, 'confidence': confidence, 'length': len(beam_text)}

                # Try argmax collapse
                arg_text = self._argmax_collapse(predictions)
                if arg_text and len(arg_text) == self.captcha_length:
                    print(f"✓ Argmax collapse success: {arg_text}")
                    return {'success': True, 'text': arg_text, 'confidence': confidence, 'length': len(arg_text)}

                # If none produced correct length, return best available (truncate/pad)
                best_text = predicted_text
                if len(best_text) < self.captcha_length:
                    # pad with placeholder (could be random or 'X')
                    best_text = best_text.ljust(self.captcha_length, 'X')
                else:
                    best_text = best_text[:self.captcha_length]

                return {
                    'success': True,
                    'text': best_text,
                    'confidence': confidence,
                    'length': len(best_text)
                }

            return {
                'success': True,
                'text': predicted_text,
                'confidence': confidence,
                'length': len(predicted_text)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': None,
                'confidence': 0.0
            }
    
    def recognize_from_base64(self, base64_string):
        """
        Recognize CAPTCHA from base64 encoded image
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            dict with recognition results
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_string)
            
            # Recognize
            return self.recognize(image_bytes)
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Base64 decode error: {str(e)}',
                'text': None,
                'confidence': 0.0
            }


class CaptchaSolver:
    """
    Small compatibility wrapper used by other modules as a fallback solver.
    If the AI recognizer is available it will use it, otherwise it will try
    an optional pytesseract fallback (only if installed). Returns an empty
    string on failure so callers can continue with other fallbacks.
    """
    def __init__(self):
        # Reuse the global recognizer if present
        try:
            self.recognizer = captcha_recognizer if 'captcha_recognizer' in globals() and captcha_recognizer.model_loaded else None
        except Exception:
            self.recognizer = None

    def solve(self, image_bytes):
        """Return a 6-char (or configured length) text string or empty string."""
        # Try AI recognizer first
        try:
            if self.recognizer:
                res = self.recognizer.recognize(image_bytes)
                if res.get('success') and res.get('text'):
                    return res.get('text')
        except Exception:
            pass

        # Optional OCR fallback using pytesseract if available
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        try:
            import pytesseract
        except Exception:
            # pytesseract not installed -> save image for debugging and return empty
            try:
                ts = int(time.time() * 1000)
                debug_path = os.path.join(logs_dir, f'captcha_no_ocr_{ts}.png')
                with open(debug_path, 'wb') as f:
                    f.write(image_bytes)
                print(f"⚠️  pytesseract not available; saved captcha to {debug_path}")
            except Exception:
                pass
            return ''

        # If pytesseract available, try a few preprocessing variants
        try:
            pil = Image.open(io.BytesIO(image_bytes)).convert('L')
        except Exception:
            return ''

        # Build whitelist from recognizer if possible
        whitelist = None
        try:
            if self.recognizer and getattr(self.recognizer, 'characters', None):
                whitelist = ''.join(self.recognizer.characters)
        except Exception:
            whitelist = None

        def try_ocr(img, psm=7, whitelist_chars=None):
            cfg = f'--psm {psm}'
            if whitelist_chars:
                # configure tesseract whitelist
                cfg += f" -c tessedit_char_whitelist={whitelist_chars}"
            try:
                return pytesseract.image_to_string(img, config=cfg)
            except Exception:
                try:
                    return pytesseract.image_to_string(img)
                except Exception:
                    return ''

        variants = []
        # original resized to reasonable width for OCR
        variants.append(pil.resize((max(100, pil.width * 2), max(30, pil.height * 2))))
        # thresholded
        variants.append(variants[0].point(lambda p: 255 if p > 128 else 0))
        # inverted
        variants.append(Image.fromarray(255 - np.array(variants[0], dtype=np.uint8)))

        # try variants with whitelist first
        for img_variant in variants:
            raw = try_ocr(img_variant, psm=7, whitelist_chars=whitelist)
            cleaned = ''.join(re.findall(r'[A-Za-z0-9]+', raw))
            if cleaned:
                return cleaned[: (self.recognizer.captcha_length if self.recognizer else 6)]

        # try without whitelist and different psm modes
        for psm in (7, 8, 6):
            raw = try_ocr(variants[0], psm=psm, whitelist_chars=None)
            cleaned = ''.join(re.findall(r'[A-Za-z0-9]+', raw))
            if cleaned:
                return cleaned[: (self.recognizer.captcha_length if self.recognizer else 6)]

        # All OCR attempts failed - save debug image and return empty
        try:
            ts = int(time.time() * 1000)
            debug_path = os.path.join(logs_dir, f'captcha_failed_{ts}.png')
            pil.save(debug_path)
            print(f"✗ OCR failed for captcha; saved debug image to {debug_path}")
        except Exception:
            pass

        return ''


# ============================================================================
# INITIALIZE CAPTCHA RECOGNIZER
# ============================================================================

# Default model paths
# Models should be placed in: backend/models/
# After training, copy these files from train output:
#   - captcha_model_best.keras
#   - model_config.json
#   - char_to_int.json
#   - int_to_char.json
# NOTE: For Windows deployment, ensure paths use forward slashes or os.path.join
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)  # Go up from backend/python to backend
MODEL_DIR = os.path.join(backend_dir, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'captcha_model_best.keras')
CONFIG_PATH = os.path.join(MODEL_DIR, 'model_config.json')

# Initialize recognizer (will gracefully fail if model not available)
captcha_recognizer = CAPTCHARecognizer(MODEL_PATH, CONFIG_PATH)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with model status"""
    return jsonify({
        'status': 'ok',
        'message': 'VTU Results API is running',
        'captcha_model_loaded': captcha_recognizer.model_loaded,
        'model_accuracy': captcha_recognizer.config.get('char_accuracy', 0) * 100 if captcha_recognizer.config else 0
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded CAPTCHA model"""
    if not captcha_recognizer.model_loaded:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    return jsonify({
        'success': True,
        'model_info': {
            'img_height': captcha_recognizer.img_height,
            'img_width': captcha_recognizer.img_width,
            'captcha_length': captcha_recognizer.captcha_length,
            'num_characters': len(captcha_recognizer.characters),
            'character_set': captcha_recognizer.characters,
            'char_accuracy': captcha_recognizer.config.get('char_accuracy', 0) * 100,
            'full_accuracy': captcha_recognizer.config.get('full_accuracy', 0) * 100
        }
    })


@app.route('/api/recognize-captcha', methods=['POST'])
def recognize_captcha():
    """
    Recognize CAPTCHA from image
    
    Request body:
        {
            "image": "base64_encoded_image_string"
        }
    """
    if not captcha_recognizer.model_loaded:
        return jsonify({
            'success': False,
            'error': 'CAPTCHA recognition model not available'
        }), 503
    
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Recognize CAPTCHA
        result = captcha_recognizer.recognize_from_base64(data['image'])
        
        if result['success']:
            return jsonify({
                'success': True,
                'captcha_text': result['text'],
                'confidence': result['confidence'],
                'length': result['length']
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Recognition failed')
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/exams', methods=['GET'])
def get_exams():
    """Get list of available exams"""
    try:
        fetcher = VTUResultsFetcher(headless=True)
        fetcher.setup_driver()
        
        exams = fetcher.get_available_exams()
        
        fetcher.close()
        
        return jsonify({
            'success': True,
            'exams': exams
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in /api/exams: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'details': error_details
        }), 500


@app.route('/api/check-exam', methods=['POST'])
def check_exam():
    """Check if an exam requires type/branch selection"""
    try:
        data = request.get_json()
        exam_url = data.get('exam_url')
        
        fetcher = VTUResultsFetcher(headless=True)
        fetcher.setup_driver()
        
        # Navigate to exam
        if not fetcher.select_exam(exam_url):
            fetcher.close()
            return jsonify({'success': False, 'error': 'Failed to navigate to exam'}), 500
        
        # Check page type
        page_type = fetcher.check_page_type()
        
        if page_type == 'exam_type_selection':
            # Get types and branches
            types_and_branches = fetcher.get_exam_types_and_branches()
            
            fetcher.close()
            return jsonify({
                'success': True,
                'requires_selection': True,
                'types': types_and_branches['types'],
                'branches': types_and_branches['branches']
            })
        elif page_type == 'usn_input':
            # Direct USN page
            fetcher.close()
            return jsonify({'success': True, 'requires_selection': False})
        else:
            fetcher.close()
            return jsonify({'success': False, 'error': 'Unknown page type'}), 500
            
    except Exception as e:
        import traceback
        print(f"Error in /api/check-exam: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/validate-usn', methods=['POST'])
def validate_usn():
    """Validate USN format"""
    data = request.get_json()
    usn = data.get('usn', '').strip().upper()
    
    # USN format: #@@##@@###
    usn_pattern = r'^\d[A-Z]{2}\d{2}[A-Z]{2}\d{3}$'
    
    is_valid = bool(re.match(usn_pattern, usn))
    
    return jsonify({
        'valid': is_valid,
        'usn': usn,
        'message': 'Valid USN format' if is_valid else 'Invalid USN format. Expected: #@@##@@### (e.g., 4PA23CS102)'
    })


@app.route('/api/fetch-results', methods=['POST'])
def fetch_results():
    """
    Fetch results for a given USN and exam with AI CAPTCHA recognition
    
    Request body:
        {
            "usn": "1XX20XX999",
            "exam_url": "exam_url_string",
            "exam_type": "lgm-1" (optional),
            "fetch_details": true (optional),
            "download": false (optional),
            "use_ai_captcha": true (optional, default: true if model loaded)
        }
    """
    try:
        data = request.get_json()
        usn = data.get('usn', '').strip().upper()
        exam_url = data.get('exam_url')
        exam_type = data.get('exam_type')
        fetch_details = data.get('fetch_details', True)
        download = data.get('download', False)
        use_ai_captcha = data.get('use_ai_captcha', captcha_recognizer.model_loaded)
        
        # Validate USN
        usn_pattern = r'^\d[A-Z]{2}\d{2}[A-Z]{2}\d{3}$'
        if not re.match(usn_pattern, usn):
            return jsonify({
                'success': False,
                'error': 'Invalid USN format'
            }), 400
        
        # Initialize fetcher with AI CAPTCHA if available
        if use_ai_captcha and captcha_recognizer.model_loaded:
            fetcher = VTUResultsFetcher(
                headless=True,
                captcha_recognizer=captcha_recognizer
            )
        else:
            fetcher = VTUResultsFetcher(headless=True)
        
        fetcher.setup_driver()
        
        # Navigate to exam
        if not fetcher.select_exam(exam_url):
            fetcher.close()
            return jsonify({
                'success': False,
                'error': 'Failed to navigate to exam page'
            }), 500
        
        # Check page type
        page_type = fetcher.check_page_type()
        
        # Handle exam type/branch selection
        if page_type == 'exam_type_selection' and exam_type:
            if not fetcher.select_first_branch_in_type(exam_type):
                fetcher.close()
                return jsonify({
                    'success': False,
                    'error': 'Failed to navigate to exam page'
                }), 500
            page_type = fetcher.check_page_type()
        
        # If still at exam type selection
        if page_type == 'exam_type_selection':
            types_and_branches = fetcher.get_exam_types_and_branches()
            fetcher.close()
            return jsonify({
                'success': False,
                'requires_selection': True,
                'types': types_and_branches['types']
            })
        
        # If another exam selection is required
        if page_type == 'exam_selection':
            sub_exams = fetcher.get_available_exams()
            fetcher.close()
            return jsonify({
                'success': False,
                'requires_sub_exam': True,
                'sub_exams': sub_exams
            })
        
        # Submit form with AI CAPTCHA recognition
        submit_result = fetcher.submit_usn_and_captcha(usn)
        
        if 'error' in submit_result:
            fetcher.close()
            return jsonify({
                'success': False,
                'error': submit_result['error'],
                'message': submit_result.get('message', ''),
                'captcha_attempts': submit_result.get('attempts', 0)
            }), 400
        
        # Extract results
        results = fetcher.extract_results(fetch_details=fetch_details)
        
        if not results:
            fetcher.close()
            return jsonify({
                'success': False,
                'error': 'Failed to extract results'
            }), 500
        
        # Download if requested
        if download:
            fetcher.download_result()
        
        fetcher.close()
        
        return jsonify({
            'success': True,
            'results': results,
            'ai_captcha_used': use_ai_captcha and captcha_recognizer.model_loaded,
            'captcha_confidence': submit_result.get('captcha_confidence', 0.0)
        })
        
    except Exception as e:
        import traceback
        print(f"Error in /api/fetch-results: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/select-branch', methods=['POST'])
def select_branch():
    """Select a branch and get the next level of exams"""
    try:
        data = request.get_json()
        exam_url = data.get('exam_url')
        branch_url = data.get('branch_url')
        
        fetcher = VTUResultsFetcher(headless=True)
        fetcher.setup_driver()
        
        # Navigate to main exam
        if not fetcher.select_exam(exam_url):
            fetcher.close()
            return jsonify({
                'success': False,
                'error': 'Failed to navigate to exam'
            }), 500
        
        # Click the branch panel
        if not fetcher.select_branch(branch_url):
            fetcher.close()
            return jsonify({
                'success': False,
                'error': 'Failed to select branch'
            }), 500
        
        # Check what page we're on now
        page_type = fetcher.check_page_type()
        
        if page_type == 'exam_selection':
            sub_exams = fetcher.get_available_exams()
            fetcher.close()
            return jsonify({
                'success': True,
                'requires_final_selection': True,
                'exams': sub_exams
            })
        elif page_type == 'usn_input':
            fetcher.close()
            return jsonify({
                'success': True,
                'ready': True
            })
        else:
            fetcher.close()
            return jsonify({
                'success': False,
                'error': 'Unknown page type'
            }), 500
            
    except Exception as e:
        import traceback
        print(f"Error in /api/select-branch: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/sub-exams', methods=['POST'])
def get_sub_exams():
    """Get sub-exams for a selected exam"""
    try:
        data = request.get_json()
        exam_url = data.get('exam_url')
        
        fetcher = VTUResultsFetcher(headless=True)
        fetcher.setup_driver()
        
        # Navigate to exam
        fetcher.select_exam(exam_url)
        
        # Get sub-exams
        sub_exams = fetcher.get_available_exams()
        
        fetcher.close()
        
        return jsonify({
            'success': True,
            'sub_exams': sub_exams
        })
        
    except Exception as e:
        import traceback
        print(f"Error in /api/sub-exams: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("VTU Results Fetcher API with AI CAPTCHA Recognition")
    print("="*70)
    print(f"Server starting on http://localhost:5000")
    print(f"CAPTCHA Model Status: {'✓ LOADED' if captcha_recognizer.model_loaded else '✗ NOT LOADED'}")
    if captcha_recognizer.model_loaded:
        print(f"Model Accuracy: {captcha_recognizer.config.get('char_accuracy', 0)*100:.2f}%")
    print("="*70)
    print()
    
    # Run server
    app.run(debug=False, host='0.0.0.0', port=5000)