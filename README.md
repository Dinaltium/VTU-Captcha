# VTU Results Fetcher

A comprehensive web application to fetch and analyze VTU (Visvesvaraya Technological University) examination results with automatic captcha solving and detailed performance analysis.

## 🌟 Features

- **Automated Result Fetching**: Selenium-based automation to fetch results from results.vtu.ac.in
- **Smart Captcha Solving**: Uses trained ML model (Keras/TensorFlow) to solve captchas automatically
- **Exam Selection**: Automatically fetches available exams and supports multi-level exam selection
- **Detailed Analysis**:
  - SGPA Calculation
  - Total marks calculation
  - Pass/Fail subject count
  - Color-coded marks display
  - Performance insights
- **Modern UI**: Built with React, shadcn/ui, and Tailwind CSS
- **Beautiful Background**: Silk gradient background inspired by Reactbits
- **USN Validation**: Client-side validation with proper format checking

## 📁 Project Structure

```
VTU/
├── captcha_solver.py          # ML-based captcha solver
├── sgpa_calculator.py         # SGPA calculation logic
├── vtu_results_fetcher.py     # Main Selenium automation script
├── api.py                     # Flask REST API backend
├── requirements.txt           # Python dependencies
├── captcha_model_best.h5      # Trained captcha model
├── captcha_model.keras        # Alternative model format
├── char_to_int.json           # Character mapping for captcha
└── frontend/                  # React frontend
    ├── src/
    │   ├── components/
    │   │   ├── ui/            # shadcn UI components
    │   │   └── SilkBackground.tsx
    │   ├── lib/
    │   │   └── utils.ts
    │   ├── App.tsx            # Main React component
    │   ├── main.tsx
    │   └── index.css
    ├── package.json
    ├── tailwind.config.js
    ├── vite.config.ts
    └── index.html
```

## 🚀 Installation & Setup

### Linux/WSL Quick Setup

**First Time Setup:**
```bash
# 1. Activate virtual environment
cd ~/captcha
source ~/tfenv/bin/activate

# 2. Fix dependencies (pytesseract version fix)
pip install pytesseract==0.3.13
pip install -r Scripts/requirements.txt

# 3. Clean Windows metadata files (if any)
find backend/models -name "*Zone.Identifier" -delete

# 4. Setup frontend (if Node.js 18+ not active)
source ~/.nvm/nvm.sh  # If using nvm
nvm use --lts
cd frontend && npm install && cd ..

# 5. Start application
chmod +x start.sh
./start.sh
```

**Note:** Your models are already in `backend/models/` - no need to copy!

### Prerequisites

- **Python 3.10+** with virtual environment
- **Node.js 18+** (install via nvm: `nvm install --lts`)
- **Chrome/Chromium** browser
- **ChromeDriver** (auto-downloaded by setup.sh)

### Manual Setup

**Backend:**
```bash
cd ~/captcha
source ~/tfenv/bin/activate
pip install -r Scripts/requirements.txt
```

**Frontend:**
```bash
cd ~/captcha/frontend
npm install
```

**Model Files:**
```bash
mkdir -p backend/models
cp models/captcha_model_best.keras backend/models/
cp models/model_config.json backend/models/
cp models/char_to_int.json backend/models/
cp models/int_to_char.json backend/models/
```

## 📖 Usage

### Quick Start

```bash
./start.sh  # Starts both backend and frontend
```

Access:
- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:5000

### Manual Start

1. **Start both servers:**
   - Backend: `cd backend/python && python api.py`
   - Frontend: `cd frontend && npm run dev`

2. **Open your browser:**
   - Navigate to `http://localhost:5173`

3. **Select an exam:**
   - Choose from the list of available exams

4. **Enter your USN:**
   - Format: `#@@##@@###` (e.g., `4PA23CS102`)
   - # = number, @ = alphabet

5. **Fetch results:**
   - Click "Fetch Results" button
   - Wait for automation to complete (usually 10-30 seconds)

6. **View results:**
   - See detailed subject-wise results
   - View SGPA and performance analysis
   - Check color-coded marks

### Using Python Script Directly

```cmd
python vtu_results_fetcher.py
```

Follow the prompts to:
1. Select an exam
2. Enter USN
3. View results in terminal

## 🎯 USN Format

VTU USN follows the format: `#@@##@@###`

Where:
- `#` = Digit (0-9)
- `@` = Letter (A-Z)

### Examples:
- `4PA23CS102` - 4th year, PA college, 2023 batch, CS branch, roll 102
- `2MN25EC021` - 2nd year, MN college, 2025 batch, EC branch, roll 021
- `4JK19IC009` - 4th year, JK college, 2019 batch, IC branch, roll 009

### Valid College Codes:
Refer to `ilide.info-usn-number-of-vtu-colleges-pr_*.pdf` for complete list of college codes.

## 🧪 API Endpoints

### GET `/api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "VTU Results API is running"
}
```

### GET `/api/exams`
Get list of available exams.

**Response:**
```json
{
  "success": true,
  "exams": [
    {
      "id": 1,
      "name": "June/July-2025 Examination",
      "url": "indexJJ25.php"
    }
  ]
}
```

### POST `/api/validate-usn`
Validate USN format.

**Request:**
```json
{
  "usn": "4PA23CS102"
}
```

**Response:**
```json
{
  "valid": true,
  "usn": "4PA23CS102",
  "message": "Valid USN format"
}
```

### POST `/api/fetch-results`
Fetch results for a USN.

**Request:**
```json
{
  "usn": "4PA23CS102",
  "exam_url": "indexJJ25.php",
  "fetch_details": true,
  "download": false
}
```

**Response:**
```json
{
  "success": true,
  "results": {
    "student_info": {
      "usn": "4PA23CS102",
      "name": "Student Name"
    },
    "semester": 4,
    "subjects": [...],
    "analysis": {
      "sgpa": 8.75,
      "total_marks": 525,
      "total_subjects": 6,
      "passed_subjects": 6,
      "failed_subjects": 0,
      "performance_message": "🎉 Excellent Performance! Well done!"
    }
  }
}
```

## 🎨 Color Coding

Marks are color-coded for easy visualization:

- **🟢 Green (90-100)**: Outstanding
- **🔵 Blue (80-89)**: Excellent
- **🟣 Purple (70-79)**: Very Good
- **🟠 Amber (60-69)**: Good
- **🟠 Orange (50-59)**: Average
- **🟡 Yellow (40-49)**: Pass
- **🔴 Red (<40)**: Fail

## 🔧 Troubleshooting

### Backend Issues

**Chrome driver not found:**
```cmd
pip install webdriver-manager
```

**TensorFlow errors:**
```cmd
pip install --upgrade tensorflow
```

**Module import errors:**
```cmd
pip install -r requirements.txt --force-reinstall
```

### Frontend Issues

**Dependencies not installing:**
```cmd
npm install --legacy-peer-deps
```

**Build errors:**
```cmd
npm run build
```

**Port already in use:**
- Change port in `vite.config.ts`:
  ```typescript
  export default defineConfig({
    server: { port: 3000 }
  })
  ```

### Captcha Solving Issues

If captcha solving fails repeatedly:
1. Check model file exists: `captcha_model_best.h5`
2. Verify `char_to_int.json` is valid
3. Model may need retraining with updated captchas

### Results Page Timeout

If the page times out (2-5 minutes after results load):
- This is expected VTU behavior
- Results are already fetched and displayed
- No action needed from user

## 📝 Important Notes

1. **VTU Website Availability**: The script depends on results.vtu.ac.in being accessible
2. **Captcha Changes**: If VTU changes captcha format, model needs retraining
3. **Rate Limiting**: Avoid making too many requests in short time
4. **Browser Visibility**: Set `headless=False` for debugging
5. **Session Timeout**: Results page auto-redirects after 2-5 minutes

## 🛠️ Development

### Building for Production

**Frontend:**
```cmd
cd frontend
npm run build
```

**Backend:**
- Use gunicorn or waitress for production:
```cmd
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

### Customization

**Change API URL:**
Edit `frontend/src/App.tsx`:
```typescript
const API_BASE_URL = 'http://your-api-url.com/api';
```

**Modify Theme:**
Edit `frontend/src/index.css` to change color scheme.

**Adjust SGPA Calculation:**
Edit `sgpa_calculator.py` grade mappings.

## 📄 License

This project is for educational purposes. Use responsibly and in accordance with VTU's terms of service.

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. Code is well-documented
2. Test thoroughly before submitting
3. Follow existing code style
4. Update README if adding features

## ⚠️ Disclaimer

This tool is created for educational purposes to help students access their results more conveniently. Users are responsible for complying with VTU's terms of service and applicable laws.

## 📞 Support

For issues:
1. Check troubleshooting section
2. Verify all dependencies are installed
3. Ensure backend and frontend are both running
4. Check browser console for errors

---

**Made with ❤️ for VTU Students**
