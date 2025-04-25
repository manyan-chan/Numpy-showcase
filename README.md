# NumPy Showcase - Streamlit App

This interactive web application, built with Streamlit and NumPy, serves as a practical demonstration of proficiency in various core concepts of the NumPy library. It allows users to explore and experiment with NumPy features in real-time through a user-friendly interface.

## Purpose

The primary goal of this app is to showcase expertise in NumPy by providing clear explanations and interactive examples for fundamental operations, including:

*   Array creation and manipulation
*   Indexing and slicing techniques
*   Mathematical operations and universal functions (ufuncs)
*   Broadcasting mechanisms
*   Statistical computations
*   Linear algebra operations
*   Saving and loading arrays

## Features

The app is divided into logical sections accessible via the sidebar:

*   **🌟 Introduction & Basics:** Overview of NumPy and the `ndarray` object.
*   **🛠️ Array Creation Techniques:** Interactive creation using `zeros`, `ones`, `arange`, `linspace`, `random`, `eye`, `full`.
*   **🔪 Indexing & Slicing Mastery:** Examples of basic slicing, integer indexing, boolean indexing, and fancy indexing, highlighting view vs. copy distinctions.
*   **➕ Mathematical Operations & Ufuncs:** Demonstrations of element-wise arithmetic, scalar operations, ufuncs (`sin`, `exp`), and matrix multiplication.
*   **📡 Broadcasting Explained:** Interactive tool to test broadcasting rules with different array shapes.
*   **📊 Statistical Power:** Usage of functions like `min`, `max`, `mean`, `median`, `std`, `var`, `percentile` with axis control.
*   **📐 Linear Algebra Operations:** Examples using `np.linalg` for determinant, inverse, eigenvalues/vectors, and solving linear equations.
*   **⚙️ Reshaping & Manipulation:** Covers `reshape`, `flatten`, `ravel`, transpose (`.T`), `concatenate`, and `stack`.
*   **💾 Saving & Loading Arrays:** Demonstrates saving/loading single arrays (`.npy`), multiple arrays (`.npz`, `.npz` compressed), and text files (`.txt`).


## Requirements

*   Python (3.7+ Recommended)
*   Streamlit
*   NumPy

## Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone https://github.com/manyan-chan/Numpy-showcase.git
    cd Numpy-showcase
2.  **Install the required libraries:**
    ```bash
    pip install streamlit numpy
    ```

## Usage

1.  Navigate to the directory containing the `numpy_showcase_app.py` file in your terminal.
2.  Run the Streamlit app:
    ```bash
    streamlit run numpy_showcase_app.py
    ```
3.  The application will automatically open in your default web browser. Interact with the sidebar to explore different NumPy concepts.

## File Structure

*   `numpy_showcase_app.py`: The main Python script containing the Streamlit application code.
*   `README.md`: This file, providing information about the application.
