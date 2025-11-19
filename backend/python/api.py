"""
Flask Backend API for VTU Results Fetcher
Provides REST endpoints for the React frontend
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vtu_results_fetcher import VTUResultsFetcher
from captcha_solver import captcha_recognizer
import threading
import re
import time

app = Flask(__name__)
CORS(app)

# Store active sessions
active_sessions = {}


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'VTU Results API is running'})


@app.route('/api/exams', methods=['GET'])
def get_exams():
    """Get list of available exams"""
    try:
        fetcher = VTUResultsFetcher(headless=True, captcha_recognizer=captcha_recognizer)
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
        
        fetcher = VTUResultsFetcher(headless=True, captcha_recognizer=captcha_recognizer)
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
            
            # Debug: Print branches data
            print("\n" + "="*70)
            print("BRANCHES DATA BEING SENT TO FRONTEND:")
            print("="*70)
            import json
            for type_id, branch_list in types_and_branches['branches'].items():
                print(f"\nType: {type_id}")
                for i, branch in enumerate(branch_list):
                    print(f"  {i+1}. {branch['name']}")
                    print(f"     URL: {branch['url']}")
            print("="*70 + "\n")
            
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
    """Fetch results for a given USN and exam"""
    try:
        data = request.get_json()
        usn = data.get('usn', '').strip().upper()
        exam_url = data.get('exam_url')
        exam_type = data.get('exam_type')  # e.g., 'lgm-1' for CBCS or 'lgm-5' for CBCS-RV
        fetch_details = data.get('fetch_details', True)
        download = data.get('download', False)
        
        # Validate USN
        usn_pattern = r'^\d[A-Z]{2}\d{2}[A-Z]{2}\d{3}$'
        if not re.match(usn_pattern, usn):
            return jsonify({
                'success': False,
                'error': 'Invalid USN format'
            }), 400
        
        # Initialize fetcher
        fetcher = VTUResultsFetcher(headless=True, captcha_recognizer=captcha_recognizer)
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
        print(f"🔍 Page type: {page_type}, exam_type: {exam_type}")
        
        # If exam type/branch selection is required AND user provided exam_type
        if page_type == 'exam_type_selection' and exam_type:
            # User has selected type - navigate to the first branch in that type
            print(f"🔍 Navigating to first branch in type: {exam_type}")
            if not fetcher.select_first_branch_in_type(exam_type):
                fetcher.close()
                return jsonify({
                    'success': False,
                    'error': 'Failed to navigate to exam page'
                }), 500
            
            # Check page type again after branch selection
            page_type = fetcher.check_page_type()
        
        # If still at exam type selection (no type provided)
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
        
        # Submit form
        submit_result = fetcher.submit_usn_and_captcha(usn)
        
        if 'error' in submit_result:
            fetcher.close()
            return jsonify({
                'success': False,
                'error': submit_result['error'],
                'message': submit_result['message']
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
            'results': results
        })
        
    except Exception as e:
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
        
        fetcher = VTUResultsFetcher(headless=True, captcha_recognizer=captcha_recognizer)
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
            # Another level of exams
            sub_exams = fetcher.get_available_exams()
            fetcher.close()
            return jsonify({
                'success': True,
                'requires_final_selection': True,
                'exams': sub_exams
            })
        elif page_type == 'usn_input':
            # Ready for USN input
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
        
        fetcher = VTUResultsFetcher(headless=True, captcha_recognizer=captcha_recognizer)
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("="*50)
    print("VTU Results Fetcher API")
    print("="*50)
    print("Server starting on http://localhost:5000")
    print("="*50)
    # Disable debug mode to avoid reloader issues during testing
    app.run(debug=False, host='0.0.0.0', port=5000)
