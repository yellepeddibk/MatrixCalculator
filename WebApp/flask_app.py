from flask import Flask, request, jsonify, send_from_directory
import numpy as np

app = Flask(__name__, static_folder='.')

# Serve the HTML file
@app.route('/')
def index():
    return send_from_directory('.', 'matrix.html')

# API endpoint for determinant calculation
@app.route('/api/determinant', methods=['POST'])
def determinant():
    matrix = request.json['matrix']
    np_matrix = np.array(matrix)
    if np_matrix.shape[0] != np_matrix.shape[1]:
        return jsonify({"error": "Matrix must be square"}), 400
    det = np.linalg.det(np_matrix)
    return jsonify({"result": round(det, 4)})

# API endpoint for RREF calculation
@app.route('/api/rref', methods=['POST'])
def rref():
    matrix = request.json['matrix']
    np_matrix = np.array(matrix)
    rref_matrix = compute_rref(np_matrix)
    return jsonify({"result": rref_matrix.tolist()})

def compute_rref(matrix):
    """Compute the Reduced Row Echelon Form (RREF) of the matrix."""
    m, n = matrix.shape
    rref = matrix.copy().astype(float)
    row = 0
    for col in range(n):
        if row >= m:
            break
        pivot = np.argmax(np.abs(rref[row:, col])) + row
        if rref[pivot, col] == 0:
            continue
        rref[[row, pivot], :] = rref[[pivot, row], :]
        rref[row, :] /= rref[row, col]
        for r in range(m):
            if r != row:
                rref[r, :] -= rref[r, col] * rref[row, :]
        row += 1
    return rref

# Serve static files (CSS and JS)
@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory('.', path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
