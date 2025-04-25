import time

import numpy as np
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="NumPy Showcase",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Helper Functions ---
def display_array_info(arr, name="Array"):
    """Displays an array and its key attributes."""
    st.write(f"**{name}:**")
    st.write(arr)
    st.write(f"* **Shape:** `{arr.shape}`")
    st.write(f"* **Data Type:** `{arr.dtype}`")
    st.write(f"* **Dimensions:** `{arr.ndim}`")
    st.write(f"* **Size (Total Elements):** `{arr.size}`")
    st.write(f"* **Memory Usage:** `{arr.nbytes} bytes`")
    st.divider()


# --- App Title and Introduction ---
st.title("🚀 NumPy Showcase 🚀")
st.markdown(
    """
Welcome! This app demonstrates proficiency in various NumPy concepts, a fundamental library for numerical computing in Python.
Explore the sidebar sections to see different features in action.
"""
)

# --- Sidebar Navigation ---
st.sidebar.header("NumPy Concepts")
options = [
    "🌟 Introduction & Basics",
    "🛠️ Array Creation Techniques",
    "🔪 Indexing & Slicing Mastery",
    "➕ Mathematical Operations & Ufuncs",
    "📡 Broadcasting Explained",
    "📊 Statistical Power",
    "📐 Linear Algebra Operations",
    "⚙️ Reshaping & Manipulation",
    "💾 Saving & Loading Arrays",
]
choice = st.sidebar.radio("Select a Concept:", options)

st.sidebar.markdown("---")
st.sidebar.info("This app uses Streamlit and NumPy.")

# --- Main Content Area ---

if choice == "🌟 Introduction & Basics":
    st.header("🌟 Introduction & Basics")
    st.markdown(
        """
        NumPy (Numerical Python) provides:
        *   A powerful N-dimensional array object (`ndarray`).
        *   Sophisticated (broadcasting) functions.
        *   Tools for integrating C/C++ and Fortran code.
        *   Useful linear algebra, Fourier transform, and random number capabilities.

        The core object is the `ndarray`, which is:
        *   **Efficient:** Fixed-size, homogeneous data type arrays are memory-efficient and allow for optimized C implementations of operations.
        *   **Convenient:** Enables vectorized operations, avoiding slow Python loops.

        Let's create a simple array and see its attributes.
        """
    )

    # Basic Array Example
    basic_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    st.subheader("Example: Basic Array")
    st.code(
        "basic_arr = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)",
        language="python",
    )
    display_array_info(basic_arr, "basic_arr")
    st.success(
        "Notice the fixed `dtype` and how attributes like `shape`, `ndim`, and `size` describe the array structure."
    )


elif choice == "🛠️ Array Creation Techniques":
    st.header("🛠️ Array Creation Techniques")
    st.markdown(
        """
        NumPy offers diverse ways to create arrays beyond `np.array()`:
        *   `np.zeros()`, `np.ones()`: For arrays filled with zeros or ones.
        *   `np.arange()`: Like Python's `range`, but returns an array.
        *   `np.linspace()`: For evenly spaced numbers over a specified interval.
        *   `np.logspace()`: For evenly spaced numbers on a log scale.
        *   `np.random`: Various functions for creating arrays with random values.
        *   `np.eye()`, `np.identity()`: For identity matrices.
        *   `np.full()`: Create an array filled with a specific value.
        """
    )

    st.subheader("Interactive Creation")
    col1, col2 = st.columns(2)

    with col1:
        create_func = st.selectbox(
            "Choose Creation Function:",
            ["zeros", "ones", "arange", "linspace", "random.rand", "eye", "full"],
        )
        dtype_str = st.selectbox(
            "Select Data Type:", ["float64", "int32", "bool"], index=0
        )
        dtype_map = {"float64": np.float64, "int32": np.int32, "bool": np.bool_}
        selected_dtype = dtype_map[dtype_str]

    with col2:
        if create_func in ["zeros", "ones", "random.rand", "full"]:
            shape_str = st.text_input(
                "Enter Shape (comma-separated, e.g., 3,4):", "3, 4"
            )
            try:
                shape_tuple = tuple(map(int, shape_str.split(",")))
            except ValueError:
                st.error("Invalid shape format. Please use comma-separated integers.")
                shape_tuple = (3, 4)  # Default
        elif create_func in ["arange", "linspace"]:
            start = st.number_input("Start:", value=0.0)
            stop = st.number_input("Stop:", value=10.0)
            if create_func == "arange":
                step = st.number_input("Step:", value=1.0)
            else:  # linspace
                num = st.number_input("Num Points:", value=50, min_value=1, step=1)
        elif create_func == "eye":
            n_dim = st.number_input("Dimension (N):", value=3, min_value=1, step=1)
            k_diag = st.number_input("Diagonal Index (k):", value=0, step=1)

    # Generate and display
    st.subheader("Result")
    code_str = "Error"
    try:
        if create_func == "zeros":
            arr = np.zeros(shape_tuple, dtype=selected_dtype)
            code_str = f"np.zeros({shape_tuple}, dtype=np.{dtype_str})"
        elif create_func == "ones":
            arr = np.ones(shape_tuple, dtype=selected_dtype)
            code_str = f"np.ones({shape_tuple}, dtype=np.{dtype_str})"
        elif create_func == "arange":
            arr = np.arange(start, stop, step, dtype=selected_dtype)
            code_str = f"np.arange({start}, {stop}, {step}, dtype=np.{dtype_str})"
        elif create_func == "linspace":
            arr = np.linspace(start, stop, int(num), dtype=selected_dtype)
            code_str = f"np.linspace({start}, {stop}, {int(num)}, dtype=np.{dtype_str})"
        elif create_func == "random.rand":
            arr = np.random.rand(*shape_tuple)  # rand only outputs float64 [0, 1)
            code_str = f"np.random.rand({', '.join(map(str, shape_tuple))})"
            st.caption(
                "Note: `np.random.rand` always produces float64 values between [0, 1). Dtype selection ignored."
            )
        elif create_func == "eye":
            arr = np.eye(n_dim, k=k_diag, dtype=selected_dtype)
            code_str = f"np.eye({n_dim}, k={k_diag}, dtype=np.{dtype_str})"
        elif create_func == "full":
            fill_val = st.number_input("Fill Value:", value=7)
            arr = np.full(shape_tuple, fill_value=fill_val, dtype=selected_dtype)
            code_str = (
                f"np.full({shape_tuple}, fill_value={fill_val}, dtype=np.{dtype_str})"
            )

        st.code(code_str, language="python")
        display_array_info(arr)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.code(code_str, language="python")  # Show attempted code even on error


elif choice == "🔪 Indexing & Slicing Mastery":
    st.header("🔪 Indexing & Slicing Mastery")
    st.markdown(
        """
        Accessing and modifying array elements efficiently is crucial. NumPy offers powerful indexing options:
        *   **Basic Slicing:** `arr[start:stop:step]`, similar to Python lists but N-dimensional.
        *   **Integer Indexing:** Accessing specific elements using their indices (`arr[row, col]`).
        *   **Boolean Indexing:** Selecting elements based on a condition (`arr[arr > 5]`).
        *   **Fancy Indexing:** Using arrays of indices to select elements (`arr[[1, 3, 5]]`).

        **Important:** Basic slicing returns a *view* (shares memory with the original), while integer and boolean/fancy indexing usually return a *copy*.
        """
    )

    # Create a sample array
    rows = st.slider("Select number of rows for sample array:", 2, 10, 5)
    cols = st.slider("Select number of columns for sample array:", 2, 10, 6)
    base_arr = np.arange(rows * cols).reshape(rows, cols)

    st.subheader("Sample Array")
    st.code(f"base_arr = np.arange({rows * cols}).reshape({rows}, {cols})")
    display_array_info(base_arr, "base_arr")

    st.subheader("Try Different Indexing Methods")

    indexing_type = st.radio(
        "Select Indexing Type:",
        [
            "Basic Slicing",
            "Integer Indexing",
            "Boolean Indexing",
            "Fancy Indexing (Rows)",
        ],
        horizontal=True,
    )

    try:
        if indexing_type == "Basic Slicing":
            st.markdown(
                "Use Python slice notation (e.g., `1:4:2`, `:3`, `::-1`). Leave blank for `:`."
            )
            row_slice_str = st.text_input("Row Slice:", "1:4")
            col_slice_str = st.text_input("Column Slice:", ":")
            code_str = f"base_arr[{row_slice_str}, {col_slice_str}]"
            st.code(code_str, language="python")

            # Need to eval the slice strings carefully
            def parse_slice(s):
                parts = s.split(":")
                return slice(*[int(p) if p.strip() else None for p in parts])

            row_slice = parse_slice(row_slice_str)
            col_slice = parse_slice(col_slice_str)
            result = base_arr[row_slice, col_slice]
            is_view = np.may_share_memory(result, base_arr)
            display_array_info(result, "Result")
            st.info(
                f"This result **{'is a view' if is_view else 'is a copy'}** of the original array."
            )

        elif indexing_type == "Integer Indexing":
            st.markdown("Select a specific element using its row and column index.")
            max_row_idx = base_arr.shape[0] - 1
            max_col_idx = base_arr.shape[1] - 1
            sel_row = st.number_input(
                f"Row Index (0 to {max_row_idx}):", 0, max_row_idx, 0
            )
            sel_col = st.number_input(
                f"Column Index (0 to {max_col_idx}):", 0, max_col_idx, 1
            )
            code_str = f"base_arr[{sel_row}, {sel_col}]"
            st.code(code_str, language="python")
            result = base_arr[sel_row, sel_col]
            st.write(f"**Result:** `{result}` (Type: `{type(result)}`)")
            # Note: Integer indexing usually returns a copy, but for single elements it returns the element itself (not an array)
            # is_view = np.may_share_memory(result, base_arr) # This would error for a scalar
            st.info(
                "Accessing a single element returns the element's value directly, not an array view or copy."
            )

        elif indexing_type == "Boolean Indexing":
            st.markdown("Select elements meeting a condition.")
            threshold = st.slider(
                "Select elements greater than:",
                int(base_arr.min()),
                int(base_arr.max()),
                int(base_arr.mean()),
            )
            condition = base_arr > threshold
            code_str = f"base_arr[base_arr > {threshold}]"

            st.write("**Condition Array (`base_arr > threshold`):**")
            st.write(condition)
            st.code(code_str, language="python")

            result = base_arr[condition]
            is_view = np.may_share_memory(result, base_arr)
            display_array_info(result, "Result")
            st.info(
                f"This result **{'is a view' if is_view else 'is a copy'}** of the original array (Boolean Indexing usually creates a copy)."
            )

        elif indexing_type == "Fancy Indexing (Rows)":
            st.markdown("Select specific rows using a list or array of indices.")
            max_row_idx = base_arr.shape[0] - 1
            default_indices = [
                0,
                min(2, max_row_idx),
                min(1, max_row_idx),
            ]  # Example default
            indices_str = st.text_input(
                f"Enter Row Indices (comma-separated, e.g., 0,2,1):",
                ",".join(map(str, default_indices)),
            )
            try:
                row_indices = list(map(int, indices_str.split(",")))
                # Basic validation
                if not all(0 <= i < base_arr.shape[0] for i in row_indices):
                    st.warning("Some indices are out of bounds.")
                code_str = f"base_arr[{row_indices}]"  # Selects entire rows
                st.code(code_str, language="python")
                result = base_arr[row_indices]
                is_view = np.may_share_memory(result, base_arr)
                display_array_info(result, "Result")
                st.info(
                    f"This result **{'is a view' if is_view else 'is a copy'}** of the original array (Fancy Indexing usually creates a copy)."
                )

            except ValueError:
                st.error("Invalid index format. Please use comma-separated integers.")

    except Exception as e:
        st.error(f"An error occurred during indexing: {e}")
        st.code(
            code_str if "code_str" in locals() else "Error in code generation",
            language="python",
        )


elif choice == "➕ Mathematical Operations & Ufuncs":
    st.header("➕ Mathematical Operations & Ufuncs")
    st.markdown(
        """
        NumPy allows element-wise operations directly on arrays, which is much faster than Python loops.
        *   **Basic Arithmetic:** `+`, `-`, `*`, `/`, `**` operate element-wise.
        *   **Universal Functions (Ufuncs):** Functions that operate element-wise on ndarrays (e.g., `np.sin()`, `np.exp()`, `np.sqrt()`, `np.add()`). They are highly optimized.
        """
    )

    # Create sample arrays
    st.subheader("Sample Arrays")
    arr_a = np.array([[1, 2], [3, 4]])
    arr_b = np.array([[5, 6], [7, 8]])
    scalar = st.number_input("Scalar Value:", value=10)

    col1, col2 = st.columns(2)
    with col1:
        st.code("arr_a = np.array([[1, 2], [3, 4]])")
        display_array_info(arr_a, "arr_a")
    with col2:
        st.code("arr_b = np.array([[5, 6], [7, 8]])")
        display_array_info(arr_b, "arr_b")

    st.subheader("Perform Operations")
    operation = st.selectbox(
        "Select Operation:",
        [
            "Addition (a + b)",
            "Multiplication (a * b)",
            "Scalar Addition (a + scalar)",
            "Sine (np.sin(a))",
            "Exponential (np.exp(a))",
            "Matrix Multiplication (a @ b)",
        ],
    )

    try:
        if operation == "Addition (a + b)":
            result = arr_a + arr_b
            code_str = "result = arr_a + arr_b"
        elif operation == "Multiplication (a * b)":
            result = arr_a * arr_b
            code_str = "result = arr_a * arr_b  # Element-wise"
        elif operation == "Scalar Addition (a + scalar)":
            result = arr_a + scalar
            code_str = f"result = arr_a + {scalar}"
        elif operation == "Sine (np.sin(a))":
            result = np.sin(arr_a)
            code_str = "result = np.sin(arr_a)"
        elif operation == "Exponential (np.exp(a))":
            result = np.exp(arr_a)
            code_str = "result = np.exp(arr_a)"
        elif operation == "Matrix Multiplication (a @ b)":
            if arr_a.shape[1] != arr_b.shape[0]:
                st.error(
                    f"Cannot perform matrix multiplication: Incompatible shapes {arr_a.shape} and {arr_b.shape}"
                )
                result = None
                code_str = "# Incompatible shapes for matrix multiplication"
            else:
                result = arr_a @ arr_b  # or np.dot(arr_a, arr_b)
                code_str = "result = arr_a @ arr_b # Matrix multiplication"

        st.code(code_str, language="python")
        if result is not None:
            display_array_info(result, "Result")
    except Exception as e:
        st.error(f"An error occurred: {e}")


elif choice == "📡 Broadcasting Explained":
    st.header("📡 Broadcasting Explained")
    st.markdown(
        """
        Broadcasting is a powerful NumPy mechanism that allows operations on arrays of *different* but *compatible* shapes without explicitly creating copies of data. It makes code cleaner and often faster.

        **The Rules of Broadcasting:**
        When operating on two arrays, NumPy compares their shapes element-wise, starting from the *trailing* (rightmost) dimensions:
        1.  **Compatible Dimensions:** Two dimensions are compatible if:
            *   They are equal, OR
            *   One of them is 1.
        2.  **Shape Comparison:** NumPy checks compatibility for all dimensions. If shapes have different numbers of dimensions, the shorter shape is padded with ones on its *left* side.
        3.  **Error:** If these conditions are not met, a `ValueError: operands could not be broadcast together` is raised.
        4.  **Result Shape:** The resulting array's shape has the maximum size along each dimension of the input arrays.
        5.  **Behavior:** The array with size 1 along a dimension behaves as if its data were copied along that dimension to match the other array's size.

        **Example:** `(4, 3)` + `(3,)` -> `(4, 3)` + `(1, 3)` -> Result shape `(4, 3)`
        **Example:** `(5, 1, 4)` + `(3, 1)` -> `(5, 1, 4)` + `(1, 3, 1)` -> Result shape `(5, 3, 4)`
        **Example:** `(5, 4)` + `(3,)` -> `(5, 4)` + `(1, 3)` -> **Error!** (Dim 0: 5 vs 1 okay, Dim 1: 4 vs 3 not okay)
        """
    )

    st.subheader("Interactive Broadcasting Example")

    shape1_str = st.text_input("Shape of Array 1 (e.g., 4,3):", "4, 3")
    shape2_str = st.text_input("Shape of Array 2 (e.g., 3,):", "3,")

    try:
        shape1 = tuple(map(int, shape1_str.split(",")))
        shape2 = tuple(map(int, shape2_str.split(",")))

        arr1 = (
            np.arange(np.prod(shape1)).reshape(shape1) + 1
        )  # Use 1-based values for clarity
        arr2 = np.arange(np.prod(shape2)).reshape(shape2) + 101  # Use different values

        st.write("**Array 1:**")
        st.code(f"arr1 = np.arange({np.prod(shape1)}).reshape({shape1}) + 1")
        display_array_info(arr1, "arr1")

        st.write("**Array 2:**")
        st.code(f"arr2 = np.arange({np.prod(shape2)}).reshape({shape2}) + 101")
        display_array_info(arr2, "arr2")

        st.subheader("Attempting Addition (arr1 + arr2)")
        try:
            start_time = time.time()
            result = arr1 + arr2
            end_time = time.time()
            st.success("Broadcasting Successful!")
            st.code("result = arr1 + arr2", language="python")
            display_array_info(result, "Result")
            st.info(f"Calculation took {(end_time - start_time) * 1000:.4f} ms.")
        except ValueError as e:
            st.error(f"Broadcasting Failed: {e}")
            st.code(
                "# ValueError: operands could not be broadcast together",
                language="python",
            )

    except ValueError:
        st.error("Invalid shape format. Please use comma-separated integers.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


elif choice == "📊 Statistical Power":
    st.header("📊 Statistical Power")
    st.markdown(
        """
        NumPy provides efficient functions for common statistical calculations:
        *   `np.min()`, `np.max()`, `np.argmin()`, `np.argmax()`
        *   `np.sum()`, `np.prod()`
        *   `np.mean()`, `np.median()`, `np.std()` (standard deviation), `np.var()` (variance)
        *   `np.percentile()`

        Many of these functions accept an `axis` argument to perform calculations along specific dimensions.
        *   `axis=None` (default): Operates on the flattened array.
        *   `axis=0`: Operates along the rows (collapses rows).
        *   `axis=1`: Operates along the columns (collapses columns).
        """
    )

    # Create sample array
    rows = st.slider("Rows for random array:", 2, 10, 4, key="stat_rows")
    cols = st.slider("Cols for random array:", 2, 10, 5, key="stat_cols")
    stat_arr = np.random.randint(0, 100, size=(rows, cols))

    st.subheader("Sample Random Array")
    st.code(f"stat_arr = np.random.randint(0, 100, size=({rows}, {cols}))")
    display_array_info(stat_arr, "stat_arr")

    st.subheader("Calculate Statistics")
    stat_func_str = st.selectbox(
        "Select Statistical Function:",
        ["min", "max", "sum", "mean", "median", "std", "var", "percentile"],
    )

    # Axis selection
    axis_options = {
        "Entire Array (axis=None)": None,
        "Along Rows (axis=0)": 0,
        "Along Columns (axis=1)": 1,
    }
    axis_label = st.radio("Select Axis:", list(axis_options.keys()), horizontal=True)
    selected_axis = axis_options[axis_label]

    # Percentile needs extra input
    q_val = 50  # Default for percentile
    if stat_func_str == "percentile":
        q_val = st.slider("Percentile (q):", 0, 100, 50)

    # Perform calculation
    try:
        stat_func = getattr(np, stat_func_str)
        if stat_func_str == "percentile":
            result = stat_func(stat_arr, q=q_val, axis=selected_axis)
            code_str = f"np.{stat_func_str}(stat_arr, q={q_val}, axis={selected_axis})"
        else:
            result = stat_func(stat_arr, axis=selected_axis)
            code_str = f"np.{stat_func_str}(stat_arr, axis={selected_axis})"

        st.code(code_str, language="python")
        st.write("**Result:**")
        if isinstance(result, np.ndarray):
            display_array_info(result, "Result")
        else:
            st.write(result)  # Scalar result
    except Exception as e:
        st.error(f"An error occurred: {e}")


elif choice == "📐 Linear Algebra Operations":
    st.header("📐 Linear Algebra Operations")
    st.markdown(
        """
        NumPy's `linalg` submodule provides essential linear algebra capabilities:
        *   **Matrix Multiplication:** `A @ B` or `np.dot(A, B)`
        *   **Determinant:** `np.linalg.det(A)`
        *   **Inverse:** `np.linalg.inv(A)` (for square, non-singular matrices)
        *   **Eigenvalues & Eigenvectors:** `np.linalg.eig(A)`
        *   **Singular Value Decomposition (SVD):** `np.linalg.svd(A)`
        *   **Solving Linear Equations:** `np.linalg.solve(A, b)` for `Ax = b`
        """
    )

    # Create sample matrices (ensure square for some ops)
    st.subheader("Sample Matrices")
    dim = st.slider("Matrix Dimension (n x n):", 2, 5, 3, key="linalg_dim")

    # Make matrices slightly non-trivial
    matrix_a = np.random.randint(1, 10, size=(dim, dim))
    matrix_b = np.random.randint(1, 10, size=(dim, dim))
    vector_b = np.random.randint(1, 10, size=dim)

    col1, col2 = st.columns(2)
    with col1:
        st.code(f"matrix_a = np.random.randint(1, 10, size=({dim}, {dim}))")
        display_array_info(matrix_a, "matrix_a")
        st.code(f"vector_b = np.random.randint(1, 10, size={dim})")
        display_array_info(vector_b, "vector_b")
    with col2:
        st.code(f"matrix_b = np.random.randint(1, 10, size=({dim}, {dim}))")
        display_array_info(matrix_b, "matrix_b")

    st.subheader("Perform Linear Algebra Operation")
    linalg_op = st.selectbox(
        "Select Operation:",
        [
            "Matrix Multiplication (A @ B)",
            "Determinant (det(A))",
            "Inverse (inv(A))",
            "Eigenvalues/vectors (eig(A))",
            "Solve Ax = b",
        ],
    )

    try:
        if linalg_op == "Matrix Multiplication (A @ B)":
            result = matrix_a @ matrix_b
            code_str = "result = matrix_a @ matrix_b"
            st.code(code_str, language="python")
            display_array_info(result, "Result (A @ B)")

        elif linalg_op == "Determinant (det(A))":
            result = np.linalg.det(matrix_a)
            code_str = "result = np.linalg.det(matrix_a)"
            st.code(code_str, language="python")
            st.write(f"**Result (Determinant):** `{result:.4f}`")
            if np.isclose(result, 0):
                st.warning(
                    "Determinant is close to zero; matrix may be singular (non-invertible)."
                )

        elif linalg_op == "Inverse (inv(A))":
            code_str = "result = np.linalg.inv(matrix_a)"
            st.code(code_str, language="python")
            try:
                result = np.linalg.inv(matrix_a)
                display_array_info(result, "Result (Inverse of A)")
                # Verification
                identity_check = matrix_a @ result
                st.write(
                    "**Verification (A @ A_inv):** (Should be close to Identity Matrix)"
                )
                st.write(identity_check)
                if np.allclose(identity_check, np.eye(dim)):
                    st.success("Inverse verified successfully!")
                else:
                    st.warning(
                        "Verification check deviates significantly from identity."
                    )
            except np.linalg.LinAlgError:
                st.error("Matrix A is singular and cannot be inverted.")

        elif linalg_op == "Eigenvalues/vectors (eig(A))":
            code_str = "eigenvalues, eigenvectors = np.linalg.eig(matrix_a)"
            st.code(code_str, language="python")
            eigenvalues, eigenvectors = np.linalg.eig(matrix_a)
            st.write("**Eigenvalues:**")
            st.write(eigenvalues)
            st.write("**Eigenvectors (each column is an eigenvector):**")
            display_array_info(eigenvectors, "Eigenvectors")

        elif linalg_op == "Solve Ax = b":
            code_str = "x = np.linalg.solve(matrix_a, vector_b)"
            st.code(code_str, language="python")
            try:
                solution_x = np.linalg.solve(matrix_a, vector_b)
                display_array_info(solution_x, "Solution vector x")
                # Verification
                check_b = matrix_a @ solution_x
                st.write(
                    "**Verification (A @ x):** (Should be close to original vector_b)"
                )
                st.write(check_b)
                if np.allclose(check_b, vector_b):
                    st.success("Solution verified successfully!")
                else:
                    st.warning(
                        "Verification check deviates significantly from original b."
                    )
            except np.linalg.LinAlgError:
                st.error(
                    "Matrix A is singular; the system Ax=b may have no unique solution."
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")


elif choice == "⚙️ Reshaping & Manipulation":
    st.header("⚙️ Reshaping & Manipulation")
    st.markdown(
        """
        Changing the shape and structure of arrays without changing their data is common.
        *   `reshape()`: Returns an array with a new shape (must have the same total number of elements). Can return a view or a copy.
        *   `flatten()`: Always returns a 1D *copy* of the array.
        *   `ravel()`: Returns a 1D *view* of the array whenever possible (more memory efficient than `flatten` if you don't need a copy).
        *   `transpose()` or `.T`: Permutes the dimensions (e.g., rows become columns). Returns a view.
        *   `np.concatenate()`: Joins arrays along an existing axis.
        *   `np.stack()`, `np.vstack()`, `np.hstack()`: Joins arrays along a *new* axis.
        """
    )

    # Sample Array
    st.subheader("Sample Array")
    manip_arr = np.arange(12)
    st.code("manip_arr = np.arange(12)")
    display_array_info(manip_arr, "manip_arr")

    st.subheader("Apply Manipulation")
    manip_op = st.selectbox(
        "Select Operation:",
        ["reshape", "flatten", "ravel", "transpose (.T)", "concatenate", "stack"],
    )

    try:
        if manip_op == "reshape":
            st.markdown(
                "Enter the new shape. The total number of elements must remain 12."
            )
            new_shape_str = st.text_input("New Shape (e.g., 3,4 or 2,6):", "3, 4")
            try:
                new_shape = tuple(map(int, new_shape_str.split(",")))
                if np.prod(new_shape) != manip_arr.size:
                    st.error(
                        f"Invalid shape: {new_shape} does not have {manip_arr.size} elements."
                    )
                else:
                    code_str = f"result = manip_arr.reshape({new_shape})"
                    result = manip_arr.reshape(new_shape)
                    st.code(code_str, language="python")
                    display_array_info(result, "Reshaped Array")
                    is_view = np.may_share_memory(result, manip_arr)
                    st.info(
                        f"Reshape result **{'is likely a view' if is_view else 'is a copy'}**."
                    )
            except ValueError:
                st.error("Invalid shape format.")

        elif manip_op == "flatten":
            code_str = "result = manip_arr.flatten()"
            result = (
                manip_arr.flatten()
            )  # Usually applied on multi-dimensional, but works on 1D
            st.code(code_str, language="python")
            display_array_info(result, "Flattened Array")
            is_view = np.may_share_memory(result, manip_arr)
            st.info(
                f"Flatten result **{'is a view' if is_view else 'is a copy'}** (Flatten always returns a copy)."
            )

        elif manip_op == "ravel":
            code_str = "result = manip_arr.ravel()"
            result = manip_arr.ravel()
            st.code(code_str, language="python")
            display_array_info(result, "Raveled Array")
            is_view = np.may_share_memory(result, manip_arr)
            st.info(
                f"Ravel result **{'is likely a view' if is_view else 'is a copy'}** (Ravel returns a view if possible)."
            )

        elif manip_op == "transpose (.T)":
            st.markdown(
                "First, let's reshape the array to 2D (e.g., 3x4) to make transpose meaningful."
            )
            arr_2d = manip_arr.reshape(3, 4)
            st.code("arr_2d = manip_arr.reshape(3, 4)")
            display_array_info(arr_2d, "arr_2d (3x4)")

            code_str = "result = arr_2d.T"
            result = arr_2d.T
            st.code(code_str, language="python")
            display_array_info(result, "Transposed Array")
            is_view = np.may_share_memory(result, arr_2d)
            st.info(
                f"Transpose result **{'is likely a view' if is_view else 'is a copy'}** (Transpose returns a view)."
            )

        elif manip_op in ["concatenate", "stack"]:
            st.markdown("Let's create two small arrays to demonstrate joining.")
            arr_c1 = np.array([[1, 2], [3, 4]])
            arr_c2 = np.array([[5, 6]])  # Note shape (1, 2) for concatenate axis=0

            st.code("arr_c1 = np.array([[1, 2], [3, 4]])")
            display_array_info(arr_c1, "arr_c1")
            st.code("arr_c2 = np.array([[5, 6]])")
            display_array_info(arr_c2, "arr_c2")

            axis_options = {"Axis 0": 0, "Axis 1": 1}
            sel_axis = st.radio(
                "Select Axis for Joining:",
                list(axis_options.keys()),
                horizontal=True,
                key=f"{manip_op}_axis",
            )
            join_axis = axis_options[sel_axis]

            if manip_op == "concatenate":
                code_str = (
                    f"result = np.concatenate((arr_c1, arr_c2), axis={join_axis})"
                )
                st.code(code_str, language="python")
                try:
                    # Adjust arr_c2 shape for axis=1 concatenation if needed
                    if join_axis == 1:
                        arr_c2_compat = (
                            arr_c2.T
                        )  # Make it (2, 1) to concat with (2, 2) along axis 1
                        st.info(
                            "Note: Reshaped arr_c2 to `arr_c2.T` for axis=1 concatenation compatibility."
                        )
                        result = np.concatenate((arr_c1, arr_c2_compat), axis=join_axis)
                    else:  # axis=0
                        result = np.concatenate((arr_c1, arr_c2), axis=join_axis)
                    display_array_info(result, "Concatenated Array")
                except ValueError as e:
                    st.error(
                        f"Concatenation Error: {e}. Arrays must have same shape except along the concatenation axis."
                    )

            else:  # stack
                # For stack, arrays must have the same shape
                arr_s2 = np.array([[5, 6], [7, 8]])  # Make it (2, 2) to match arr_c1
                st.code("arr_s2 = np.array([[5, 6], [7, 8]]) # Used for stacking")
                display_array_info(arr_s2, "arr_s2 (for stacking)")
                code_str = f"result = np.stack((arr_c1, arr_s2), axis={join_axis})"
                st.code(code_str, language="python")
                try:
                    result = np.stack((arr_c1, arr_s2), axis=join_axis)
                    display_array_info(result, "Stacked Array")
                    st.info(f"Notice the new dimension added at axis {join_axis}.")
                except ValueError as e:
                    st.error(f"Stacking Error: {e}. Arrays must have the same shape.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

elif choice == "💾 Saving & Loading Arrays":
    st.header("💾 Saving & Loading Arrays")
    st.markdown(
        """
        NumPy allows you to save array data to disk and load it back efficiently.
        *   `np.save('file.npy', arr)`: Saves a single array in NumPy's binary `.npy` format (efficient and preserves dtype/shape).
        *   `np.load('file.npy')`: Loads an array from an `.npy` file.
        *   `np.savez('archive.npz', arr1=arr1, arr2=arr2)`: Saves *multiple* arrays into an uncompressed `.npz` archive. Access loaded arrays like a dictionary (`loaded['arr1']`).
        *   `np.savez_compressed('archive_comp.npz', arr1=arr1, arr2=arr2)`: Saves multiple arrays into a *compressed* `.npz` archive (useful for large arrays).
        *   `np.savetxt()`, `np.loadtxt()`: For saving/loading plain text files (less efficient, loses precision/type info sometimes, but human-readable).
        """
    )

    # Create sample arrays
    st.subheader("Sample Arrays for Saving")
    arr_save1 = np.random.rand(3, 4) * 100
    arr_save2 = np.random.randint(0, 10, size=(5,), dtype=np.int16)

    col1, col2 = st.columns(2)
    with col1:
        st.code("arr_save1 = np.random.rand(3, 4) * 100")
        display_array_info(arr_save1, "arr_save1")
    with col2:
        st.code("arr_save2 = np.random.randint(0, 10, size=(5,), dtype=np.int16)")
        display_array_info(arr_save2, "arr_save2")

    # --- Saving ---
    st.subheader("Saving Arrays")
    save_format = st.radio(
        "Choose Save Format:",
        [
            ".npy (single array)",
            ".npz (multiple, uncompressed)",
            ".npz (multiple, compressed)",
            ".txt (single array)",
        ],
        horizontal=True,
    )

    file_path_base = "temp_numpy_showcase"  # Base name for files

    if st.button("Save Arrays"):
        try:
            if save_format == ".npy (single array)":
                file_path = f"{file_path_base}_single.npy"
                np.save(file_path, arr_save1)
                st.success(f"Saved `arr_save1` to `{file_path}`")
                st.session_state["last_saved_npy"] = file_path  # Store path for loading
            elif save_format == ".npz (multiple, uncompressed)":
                file_path = f"{file_path_base}_archive.npz"
                np.savez(file_path, first_array=arr_save1, second_array=arr_save2)
                st.success(
                    f"Saved `arr_save1` (as 'first_array') and `arr_save2` (as 'second_array') to `{file_path}`"
                )
                st.session_state["last_saved_npz"] = file_path
            elif save_format == ".npz (multiple, compressed)":
                file_path = f"{file_path_base}_archive_comp.npz"
                np.savez_compressed(file_path, array_a=arr_save1, array_b=arr_save2)
                st.success(
                    f"Saved `arr_save1` (as 'array_a') and `arr_save2` (as 'array_b') to compressed `{file_path}`"
                )
                st.session_state["last_saved_npz_comp"] = file_path
            elif save_format == ".txt (single array)":
                file_path = f"{file_path_base}_single.txt"
                np.savetxt(
                    file_path, arr_save1, fmt="%.4f", delimiter=","
                )  # Specify format and delimiter
                st.success(f"Saved `arr_save1` to `{file_path}` (text format)")
                st.session_state["last_saved_txt"] = file_path
        except Exception as e:
            st.error(f"Error saving file: {e}")

    # --- Loading ---
    st.subheader("Loading Arrays")
    st.markdown(
        "Attempt to load the *last saved file* of the chosen corresponding type."
    )

    load_type = st.radio(
        "Choose Type to Load:",
        [".npy", ".npz", ".npz (compressed)", ".txt"],
        horizontal=True,
        key="load_radio",
    )

    if st.button("Load Array(s)"):
        loaded_successfully = False
        try:
            if load_type == ".npy" and "last_saved_npy" in st.session_state:
                file_path = st.session_state["last_saved_npy"]
                st.write(f"Loading from: `{file_path}`")
                loaded_arr = np.load(file_path)
                st.code(f"loaded_arr = np.load('{file_path}')", language="python")
                display_array_info(loaded_arr, "Loaded Array")
                loaded_successfully = True
            elif load_type == ".npz" and "last_saved_npz" in st.session_state:
                file_path = st.session_state["last_saved_npz"]
                st.write(f"Loading from: `{file_path}`")
                loaded_data = np.load(file_path)
                st.code(f"loaded_data = np.load('{file_path}')", language="python")
                st.write("Arrays in archive:", list(loaded_data.files))
                st.code(
                    "arr1 = loaded_data['first_array']\narr2 = loaded_data['second_array']",
                    language="python",
                )
                display_array_info(loaded_data["first_array"], "Loaded 'first_array'")
                display_array_info(loaded_data["second_array"], "Loaded 'second_array'")
                loaded_data.close()  # Good practice for npz files
                loaded_successfully = True
            elif (
                load_type == ".npz (compressed)"
                and "last_saved_npz_comp" in st.session_state
            ):
                file_path = st.session_state["last_saved_npz_comp"]
                st.write(f"Loading from: `{file_path}`")
                loaded_data = np.load(file_path)
                st.code(f"loaded_data = np.load('{file_path}')", language="python")
                st.write("Arrays in archive:", list(loaded_data.files))
                st.code(
                    "arr_a = loaded_data['array_a']\narr_b = loaded_data['array_b']",
                    language="python",
                )
                display_array_info(loaded_data["array_a"], "Loaded 'array_a'")
                display_array_info(loaded_data["array_b"], "Loaded 'array_b'")
                loaded_data.close()
                loaded_successfully = True
            elif load_type == ".txt" and "last_saved_txt" in st.session_state:
                file_path = st.session_state["last_saved_txt"]
                st.write(f"Loading from: `{file_path}`")
                # Need to know the delimiter used during saving
                loaded_arr = np.loadtxt(file_path, delimiter=",")
                st.code(
                    f"loaded_arr = np.loadtxt('{file_path}', delimiter=',')",
                    language="python",
                )
                display_array_info(loaded_arr, "Loaded Array (from text)")
                st.warning(
                    "Note: Loading from text might change `dtype` (often defaults to float)."
                )
                loaded_successfully = True

            if not loaded_successfully:
                st.warning(
                    "No file of the selected type was saved in this session, or the file path is missing."
                )

        except FileNotFoundError:
            st.error(f"File not found. Was it saved in this session?")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# --- Footer ---
st.markdown("---")
st.caption("NumPy Showcase App - Demonstrating core NumPy features interactively.")
