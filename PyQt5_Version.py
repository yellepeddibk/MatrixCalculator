from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QStackedWidget, QLabel, 
                             QGridLayout, QLineEdit, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import sys
import numpy as np

class MatrixSizeInput(QWidget):
    def __init__(self, app, next_widget_class):
        super().__init__()
        self.app = app
        self.next_widget_class = next_widget_class
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.topLayout = QHBoxLayout()
        self.backButton = QPushButton("Back")
        self.backButton.setFixedSize(50, 20)
        self.closeButton = QPushButton("Close")
        self.closeButton.setFixedSize(50, 20)
        self.closeButton.clicked.connect(self.closeApp)
        self.topLayout.addWidget(self.backButton, alignment=Qt.AlignLeft)
        self.topLayout.addWidget(self.closeButton, alignment=Qt.AlignRight)
        self.layout.addLayout(self.topLayout)

        self.layout.addStretch(1)

        self.rowsLabel = QLabel("Rows:")
        self.rowsInput = QLineEdit()
        self.layout.addWidget(self.rowsLabel)
        self.layout.addWidget(self.rowsInput)

        self.colsLabel = QLabel("Columns:")
        self.colsInput = QLineEdit()
        self.layout.addWidget(self.colsLabel)
        self.layout.addWidget(self.colsInput)

        self.clearButton = QPushButton("Clear")
        self.clearButton.setFixedSize(50, 20)
        self.clearButton.clicked.connect(self.clearInputs)
        self.layout.addWidget(self.clearButton, alignment=Qt.AlignRight)

        self.errorMessage = QLabel("")  # Label to display error messages
        self.errorMessage.setStyleSheet("color: red")  # Set the text color to red
        self.layout.addWidget(self.errorMessage)

        self.layout.addStretch(2)  # Add a stretch at the bottom of the layout

        self.nextButton = QPushButton("Next")
        self.nextButton.setFixedSize(50, 20)
        self.layout.addWidget(self.nextButton, alignment=Qt.AlignRight)

        self.nextButton.clicked.connect(self.checkMatrixSize)
        self.backButton.clicked.connect(self.back)

    def checkMatrixSize(self):
        rows = self.rowsInput.text()
        cols = self.colsInput.text()
        if not rows or not cols:
            self.errorMessage.setText("Please enter values")  # Display error message in the label
        else:
            try:
                rows = int(rows)
                cols = int(cols)
                if self.next_widget_class == MatrixDeterminant and rows != cols:
                    self.errorMessage.setText("The determinant must be square. Please match rows and columns.")  # Display error message for non-square matrix
                else:
                    self.errorMessage.setText("")  # Clear the error message
                    self.app.next(self.next_widget_class, rows, cols)
            except ValueError:
                self.errorMessage.setText("Please enter valid integers")  # Display error message if values are not valid integers

    def clearInputs(self):
        self.rowsInput.clear()
        self.colsInput.clear()
        self.errorMessage.clear()

    def back(self):
        self.app.back()

    def closeApp(self):
        QApplication.quit()

class MatrixInput(QWidget):
    def __init__(self, app, rows=None, cols=None):
        super().__init__()
        self.app = app
        self.rows = rows
        self.cols = cols
        self.initUI()

    def initUI(self):
        self.outerLayout = QVBoxLayout()  # Create a QVBoxLayout
        self.topLayout = QHBoxLayout()  # Create a QHBoxLayout for the top bar
        self.innerLayout = QHBoxLayout()  # Create a QHBoxLayout for the matrix
        self.layout = QGridLayout()
        self.innerLayout.addStretch(1)
        self.innerLayout.addLayout(self.layout)
        self.innerLayout.addStretch(1)

        self.backButton = QPushButton("Back")
        self.backButton.setFixedSize(50, 20)
        self.closeButton = QPushButton("Close")
        self.closeButton.setFixedSize(50, 20)
        self.closeButton.clicked.connect(self.closeApp)
        self.topLayout.addWidget(self.backButton, alignment=Qt.AlignLeft)
        self.topLayout.addWidget(self.closeButton, alignment=Qt.AlignRight)
        self.outerLayout.addLayout(self.topLayout)

        self.outerLayout.addLayout(self.innerLayout)
        self.setLayout(self.outerLayout)  # Set the QVBoxLayout as the layout

        self.matrix = []

        # Add the calculate button
        self.calculateButton = QPushButton("Calculate")
        self.calculateButton.clicked.connect(self.calculate)  # Connect the button to the calculate method

        # Add the reset button
        self.resetButton = QPushButton("Reset")
        self.resetButton.clicked.connect(self.resetMatrix)  # Connect the button to the resetMatrix method

        # Add the result label
        self.resultLabel = QLabel("Result: ")

        # Add the error message label
        self.errorMessage = QLabel("")
        self.errorMessage.setStyleSheet("color: red")

        if self.rows and self.cols:
            self.setMatrixSize(self.rows, self.cols)

        self.backButton.clicked.connect(self.back)

    def setMatrixSize(self, rows, cols):
        self.rows = rows
        self.cols = cols

        # Clear the previous matrix
        for i in reversed(range(self.layout.count())): 
            widget = self.layout.itemAt(i).widget()
            if widget is not None:  # Check if a widget exists at the given index
                widget.setParent(None)

        self.matrix = []  # Clear the matrix list

        # Create the new matrix and add the QLineEdit widgets to the layout
        for i in range(rows):
            row = []
            for j in range(cols):
                cell = QLineEdit()
                cell.setStyleSheet("background-color: lightgrey")  # Set the background color to light grey
                row.append(cell)
                self.layout.addWidget(cell, i+1, j)  # Add the QLineEdit widget to the layout
            self.matrix.append(row)

        # Add the reset and calculate buttons
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.resetButton)
        buttonLayout.addWidget(self.calculateButton)
        self.layout.addLayout(buttonLayout, rows+1, 0, 1, cols)  # Position the buttons directly under the matrix

        # Add the result label
        self.layout.addWidget(self.resultLabel, rows+2, 0)

        # Add the error message label to the layout
        self.layout.addWidget(self.errorMessage, rows+3, 0, 1, cols)

        # Adjust the stretch factors
        self.outerLayout.setStretchFactor(self.innerLayout, 3)
        self.outerLayout.setStretchFactor(self.topLayout, 1)

    def resetMatrix(self):
        # Clear the matrix inputs
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                self.matrix[i][j].setText('')
        # Clear the result and error message labels
        self.resultLabel.setText("Result: ")
        self.errorMessage.setText("")

    def calculate(self):
        pass

    def back(self):
        self.app.back()

    def closeApp(self):
        QApplication.quit()

class MatrixDeterminant(MatrixInput):
    def calculate(self):
        try:
            # Check if the matrix is square
            if self.rows != self.cols:
                raise ValueError("Rows and columns must match for the determinant.")

            # Convert the matrix of QLineEdit objects to a numpy array of floats
            matrix = np.array([[float(self.matrix[i][j].text()) if self.matrix[i][j].text() != '' else None for j in range(len(self.matrix[i]))] for i in range(len(self.matrix))])

            # Check if any value is None
            if any(None in row for row in matrix):
                raise ValueError("Please input all values")

            # Calculate the determinant
            determinant = np.linalg.det(matrix)

            # Round the determinant to the nearest ten-thousandth
            determinant = round(determinant, 4)

            # Update the result label
            self.resultLabel.setText(f"Result: {determinant}")
            self.errorMessage.setText("")  # Clear the error message

        except ValueError as e:
            # Show an error message if the matrix is not square or cells contain non-numeric text
            self.errorMessage.setText(str(e))

class MatrixRREF(MatrixInput):
    def calculate(self):
        try:
            # Convert the matrix of QLineEdit objects to a numpy array of floats
            matrix = np.array([[float(self.matrix[i][j].text()) if self.matrix[i][j].text() != '' else None for j in range(len(self.matrix[i]))] for i in range(len(self.matrix))])

            # Check if any value is None
            if any(None in row for row in matrix):
                raise ValueError("Please input all values")

            # Calculate the RREF
            rref = self.compute_rref(matrix)

            # Format the result to show it clearly
            formatted_result = '\n'.join(['\t'.join(map(lambda x: f"{x: .4f}", row)) for row in rref])
            self.resultLabel.setText(f"Result:\n{formatted_result}")
            self.errorMessage.setText("")  # Clear the error message

        except ValueError as e:
            # Show an error message if cells contain non-numeric text
            self.errorMessage.setText(str(e))

    def compute_rref(self, matrix):
        """ Compute the Reduced Row Echelon Form (RREF) of the matrix. """
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

class HomeScreen(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.layout.addStretch(1)

        self.titleLabel = QLabel("Matrix Calculator")
        font = QFont()
        font.setPointSize(24)
        self.titleLabel.setFont(font)
        self.layout.addWidget(self.titleLabel, alignment=Qt.AlignCenter)

        self.layout.addStretch(3)

        self.rrefButton = QPushButton("RREF")
        self.rrefButton.setFixedSize(100, 50)
        self.rrefButton.setStyleSheet("QPushButton {border: 2px solid #000; border-radius: 10px;}")
        self.layout.addWidget(self.rrefButton, alignment=Qt.AlignCenter)

        self.beginButton = QPushButton("Determinant")
        self.beginButton.setFixedSize(100, 50)
        self.beginButton.setStyleSheet("QPushButton {border: 2px solid #000; border-radius: 10px;}")
        self.layout.addWidget(self.beginButton, alignment=Qt.AlignCenter)

        self.layout.addStretch(1)

        self.closeButton = QPushButton("Close")
        self.closeButton.setFixedSize(100, 50)
        self.closeButton.setStyleSheet("QPushButton {border: 2px solid #000; border-radius: 10px;}")
        self.closeButton.clicked.connect(self.closeApp)
        self.layout.addWidget(self.closeButton, alignment=Qt.AlignCenter)

    def closeApp(self):
        QApplication.quit()

class MyApp(QApplication):
    def __init__(self, sys_argv):
        super(MyApp, self).__init__(sys_argv)
        self.stack = QStackedWidget()
        self.previous_widget = None  # Track the previous widget

        self.homeScreen = HomeScreen(self)
        self.matrixSizeInputDet = MatrixSizeInput(self, MatrixDeterminant)
        self.matrixSizeInputRREF = MatrixSizeInput(self, MatrixRREF)

        self.stack.addWidget(self.homeScreen)
        self.stack.addWidget(self.matrixSizeInputDet)
        self.stack.addWidget(self.matrixSizeInputRREF)

        self.homeScreen.beginButton.clicked.connect(lambda: self.switch_to(self.matrixSizeInputDet))
        self.homeScreen.rrefButton.clicked.connect(lambda: self.switch_to(self.matrixSizeInputRREF))
        self.matrixSizeInputDet.backButton.clicked.connect(self.back)
        self.matrixSizeInputRREF.backButton.clicked.connect(self.back)

        self.stack.setGeometry(750, 150, 410, 812)
        self.stack.show()

    def switch_to(self, next_widget):
        self.previous_widget = self.stack.currentWidget()  # Update the previous widget
        self.stack.setCurrentWidget(next_widget)

    def next(self, next_page_class, rows, cols):
        next_widget = next_page_class(self, rows, cols)
        self.stack.addWidget(next_widget)
        self.previous_widget = self.stack.currentWidget()  # Update the previous widget
        self.stack.setCurrentWidget(next_widget)

    def back(self):
        if self.previous_widget:
            self.stack.setCurrentWidget(self.previous_widget)
            self.previous_widget = None  # Reset previous widget after navigating back
        else:
            self.stack.setCurrentWidget(self.homeScreen)

if __name__ == "__main__":
    app = MyApp(sys.argv)
    sys.exit(app.exec_())
