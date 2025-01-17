let operation = 'determinant';

// Set operation (determinant or RREF) and reset previous errors
function setOperation(op) {
    operation = op;
    clearErrorMessage();
    document.getElementById('result').innerText = ''; // Clear previous result

    // Highlight the selected operation
    document.getElementById('determinant-button').classList.remove('btn-selected');
    document.getElementById('rref-button').classList.remove('btn-selected');
    if (operation === 'determinant') {
        document.getElementById('determinant-button').classList.add('btn-selected');
    } else {
        document.getElementById('rref-button').classList.add('btn-selected');
    }
}

// Generate matrix input fields
function generateMatrix() {
    const rows = parseInt(document.getElementById('rows').value);
    const cols = parseInt(document.getElementById('cols').value);
    const matrixContainer = document.getElementById('matrix');
    clearErrorMessage();

    if (!rows || !cols || rows <= 0 || cols <= 0) {
        displayErrorMessage('Please enter valid matrix dimensions.');
        return;
    }

    matrixContainer.innerHTML = ''; // Clear previous matrix

    const table = document.createElement('table');
    table.classList.add('matrix-grid');

    for (let i = 0; i < rows; i++) {
        const tr = document.createElement('tr');
        for (let j = 0; j < cols; j++) {
            const td = document.createElement('td');
            const input = document.createElement('input');
            input.type = 'number';
            input.className = 'form-control matrix-cell';
            input.style.width = '60px';
            td.appendChild(input);
            tr.appendChild(td);
        }
        table.appendChild(tr);
    }

    matrixContainer.appendChild(table);
    document.getElementById('matrix-section').style.display = 'block'; // Show matrix input section
}

// Reset matrix and errors
function resetMatrix() {
    document.getElementById('matrix').innerHTML = '';
    document.getElementById('result-section').style.display = 'none';
    clearErrorMessage();
}

// Validate matrix input before sending to backend
function calculate() {
    const matrixInputs = document.querySelectorAll('.matrix-cell');
    const rows = parseInt(document.getElementById('rows').value);
    const cols = parseInt(document.getElementById('cols').value);
    const matrix = [];
    clearErrorMessage();

    // Check if all values are filled
    let allFilled = true;
    for (let i = 0; i < rows; i++) {
        const row = [];
        for (let j = 0; j < cols; j++) {
            const value = matrixInputs[i * cols + j].value;
            if (!value) {
                allFilled = false;
            }
            row.push(parseFloat(value));
        }
        matrix.push(row);
    }

    if (!allFilled) {
        displayErrorMessage('Please fill in all matrix values.');
        return;
    }

    // Check if determinant is selected and matrix is not square
    if (operation === 'determinant' && rows !== cols) {
        displayErrorMessage('Determinant can only be calculated for square matrices.');
        return;
    }

    const endpoint = operation === 'determinant' ? '/api/determinant' : '/api/rref';

    fetch(endpoint, { // endpoint is either '/api/determinant' or '/api/rref' depending on the value of operation
        method: 'POST', // Make a POST request
        headers: { 'Content-Type': 'application/json' }, // Inform the server we're sending JSON
        body: JSON.stringify({ matrix }) // Convert the matrix data to JSON and send it in the request body
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            displayErrorMessage('Error: ' + data.error);
        } else if (operation === 'determinant') {
            // Display determinant result as a number
            document.getElementById('result').innerText = 'Determinant: ' + JSON.stringify(data.result);
            document.getElementById('result-section').style.display = 'block';
        } else {
            // Display RREF result as a rectangular matrix
            displayMatrix(data.result);
        }
    })
    .catch(error => console.error('Error:', error));
}

// Display matrix result (for RREF)
function displayMatrix(matrix) {
    const resultContainer = document.getElementById('result');
    resultContainer.innerHTML = ''; // Clear previous result

    const table = document.createElement('table');
    table.classList.add('matrix-grid');

    matrix.forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(cellValue => {
            const td = document.createElement('td');
            td.innerText = formatValue(cellValue);
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    resultContainer.appendChild(table);
    document.getElementById('result-section').style.display = 'block'; // Show result section
}

// Format value to remove unnecessary decimal places
function formatValue(value) {
    if (Number.isInteger(value)) {
        return value;  // Return the value as-is if it's an integer
    } else {
        return value.toFixed(4).replace(/\.?0+$/, '');  // Round to 4 decimal places and remove trailing zeroes
    }
}

// Display error message in the UI
function displayErrorMessage(message) {
    const errorContainer = document.getElementById('error-message');
    errorContainer.innerText = message;
    errorContainer.style.display = 'block';
}

// Clear any previous error messages
function clearErrorMessage() {
    const errorContainer = document.getElementById('error-message');
    errorContainer.style.display = 'none';
    errorContainer.innerText = '';
}
