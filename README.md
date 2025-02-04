# Matrix Calculator Web App | Flask, AWS, NumPy, PyQt5, Bootstrap

## Overview
The Matrix Calculator Web App began as a simple desktop application built with PyQt5 for performing essential matrix computations. Initially designed for quick calculations of determinants and Reduced Row Echelon Form (RREF), the project evolved into a Flask-powered web application to enhance accessibility and scalability. 

The web version features a modern, responsive interface built with HTML, CSS (Bootstrap), and JavaScript. Although it was previously hosted on AWS EC2 with Nginx, Gunicorn, and SSL encryption, the AWS deployment is currently offline due to cost considerations. However, you should be able to run the Web App through your local host - instructions are below.

This project demonstrates robust input validation, comprehensive error handling, and efficient computation methods, ensuring a reliable user experience in both desktop and web environments.

---

## Features

### Desktop (PyQt5) Version
- **Standalone GUI:** A dedicated desktop application built with PyQt5.
- **Matrix Input Validation:** Ensures correct matrix values prior to computation.
- **Supported Operations:**
  - Determinant Calculation (for square matrices)
  - Reduced Row Echelon Form (RREF)
- **Error Handling:** Provides clear notifications for invalid inputs.

### Web Version (Flask & JavaScript)
- **Modern Web Interface:** Developed using HTML, CSS (Bootstrap), and JavaScript for an intuitive user experience.
- **Matrix Computation API:**
  - `/api/determinant` – Computes the determinant of a given matrix.
  - `/api/rref` – Computes the Reduced Row Echelon Form (RREF) of a given matrix.
- **Dynamic Input Fields:** Allows users to adjust matrix dimensions and input values on the fly.
- **Responsive Design:** Ensures compatibility across various devices.
- **Error Handling:** Returns informative messages for invalid or erroneous inputs.

### Deployment & Optimization (AWS - Previously Hosted)
- **Cloud Hosting:** Deployed on AWS EC2 for scalability.
- **Security:** Configured with Nginx as a reverse proxy and secured with SSL encryption.
- **Performance:** Optimized using Gunicorn to manage multiple concurrent users.
- **Maintenance:** Automated scripts monitor server health and optimize response times.
- **Status:** AWS deployment is currently offline due to cost considerations.

---

## Project Structure
```
MatrixCalculator/
├── MainWebApp/                   # Folder containing the Flask web app
│   ├── flask_app.py              # Flask backend server
│   ├── matrix.html               # Web interface (HTML file)
│   ├── matrix.js                 # JavaScript logic for the frontend
│   └── style.css                 # Styling for the web app
├── LICENSE                       # MIT License file
├── PyQt5_Version.py              # PyQt5 GUI version of the calculator
└── README.md                     # Documentation for the project (this file)
```
---

## Installation & Setup

### For the Desktop (PyQt5) Version
```bash
# Clone the repository:
git clone https://github.com/yellepeddibk/MatrixCalculator.git
cd MatrixCalculator

# Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies:
pip install -r requirements.txt

# Run the application:
python PyQt5_Version.py
```

### For the Web (Flask) Version
```bash
# Navigate to the MainWebApp directory:
cd MainWebApp

# Run the Flask server:
python flask_app.py

# Access the web app on your local machine:
http://127.0.0.1:80

# To access from another device on the same network, use the address shown in the terminal output.
```

## API Endpoints

### Determinant Calculation:
#### - Endpoint: ```/api/determinant```
#### - Method: POST
#### - Request Body (JSON):
```json
{
  "matrix": [[2, 4], [3, 1]]
}
```
#### - Response:
```json
{
  "result": -10.0
}
```

### Reduced Row Echelon Form (RREF)
#### - Endpoint: ```/api/rref```
#### - Method: POST
#### - Request Body (JSON):
```
{
  "matrix": [[1, 2], [3, 4]]
}
```
#### - Response:
```
{
  "result": [[1, 0], [0, 1]]
}
```

## Requirements
### To run this project, install the following Python libraries:

#### - Flask
#### - NumPy
#### - PyQt5
#### - Gunicorn (for production deployment)
### Other dependencies listed in ```requirements.txt```:
```bash
pip install -r requirements.txt
```

## Future Enhancements
#### - Implement additional matrix operations such as inversion, multiplication, and rank determination.
#### - Introduce advanced visualization features for matrix transformations.
#### - Explore alternative cloud deployment options to reduce hosting costs.
#### - Expand the API to handle more complex mathematical computations.

## License
#### This project is licensed under the MIT License.

## Author
### Bhargav Yellepeddi
#### - GitHub: @yellepeddibk