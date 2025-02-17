# Implementation inspired by DSA course materials

class MatrixDimensionError(Exception):
    """Custom exception for matrix dimension mismatches"""
    pass

class MatrixIndexError(Exception):
    """Custom exception for invalid matrix indices"""
    pass

class SparseMatrix:
    """
    A memory-efficient implementation of a sparse matrix using dictionary of dictionaries
    and CSR format for multiplication.
    
    Storage format: {row: {col: value}} where only non-zero elements are stored.
    
    Time Complexity:
    - Get/Set Element: O(1)
    - Addition/Subtraction: O(n) where n is number of non-zero elements
    - Multiplication: O(n*m) where n,m are non-zero elements in matrices
    - CSR Conversion: O(n) where n is number of non-zero elements
    """
    
    def __init__(self, source=None, rows=0, cols=0):
        """Initialize sparse matrix from file or dimensions"""
        self.data = {}  # {row: {col: value}}
        self.rows = rows
        self.cols = cols
        self.nnz = 0  # Number of non-zero elements
        
        if isinstance(source, str):
            self._load_from_file(source)
    
    def _load_from_file(self, file_path):
        """
        Parse matrix file and load data with strict dimension checking
        """
        try:
            with open(file_path, 'r') as f:
                # Read dimensions
                rows_line = f.readline().strip()
                cols_line = f.readline().strip()
                
                if not rows_line.startswith('rows=') or not cols_line.startswith('cols='):
                    raise ValueError("First two lines must be in format 'rows=N' and 'cols=N'")
                
                try:
                    self.rows = int(rows_line.split('=')[1])
                    self.cols = int(cols_line.split('=')[1])
                except ValueError:
                    raise ValueError("Invalid dimension values")
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if not (line.startswith('(') and line.endswith(')')):
                        raise ValueError(f"Invalid format: {line}")
                    
                    values = [v.strip() for v in line[1:-1].split(',')]
                    if len(values) != 3:
                        raise ValueError(f"Invalid element format: {line}")
                    
                    try:
                        row, col, value = map(int, values)
                        # Adjust indices if they match the dimensions exactly
                        if row == self.rows:
                            row -= 1
                        if col == self.cols:
                            col -= 1
                        
                        # Strict dimension checking
                        if row < 0 or row >= self.rows:
                            raise MatrixIndexError(f"Row index {row} out of range [0, {self.rows-1}]")
                        if col < 0 or col >= self.cols:
                            raise MatrixIndexError(f"Column index {col} out of range [0, {self.cols-1}]")
                        
                        if value != 0:
                            self.data[row] = self.data.get(row, {})
                            self.data[row][col] = value
                            self.nnz += 1
                    except ValueError:
                        raise ValueError(f"Invalid numeric values in line: {line}")
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"Matrix file not found: {file_path}")
        except ValueError as e:
            raise ValueError(f"Invalid matrix file format: {str(e)}")
    
    def to_csr(self):
        """
        Convert matrix to CSR (Compressed Sparse Row) format
        
        Returns:
            tuple: (values, col_indices, row_ptr)
        
        Time Complexity: O(n) where n is number of non-zero elements
        """
        if not self.data:
            return [], [], [0]
        
        values = []
        col_indices = []
        row_ptr = [0]
        
        count = 0
        for row in range(self.rows):
            if row in self.data:
                for col in sorted(self.data[row].keys()):
                    values.append(self.data[row][col])
                    col_indices.append(col)
                    count += 1
            row_ptr.append(count)
        
        return values, col_indices, row_ptr
    
    def add(self, other):
        """
        Add two sparse matrices with strict dimension checking
        
        Raises:
            MatrixDimensionError: If matrix dimensions don't match exactly
        
        Time Complexity: O(n) where n is total non-zero elements
        """
        if (self.rows, self.cols) != (other.rows, other.cols):
            raise MatrixDimensionError(
                f"Cannot add matrices of different dimensions: "
                f"{self.rows}x{self.cols} and {other.rows}x{other.cols}"
            )
        
        result = SparseMatrix(rows=self.rows, cols=self.cols)
        
        # Add elements from self
        for row in self.data:
            for col, value in self.data[row].items():
                result.set_element(row, col, value)
        
        # Add elements from other
        for row in other.data:
            for col, value in other.data[row].items():
                current = result.get_element(row, col)
                result.set_element(row, col, current + value)
        
        return result
    
    def subtract(self, other):
        """
        Subtract two sparse matrices with strict dimension checking
        
        Raises:
            MatrixDimensionError: If matrix dimensions don't match exactly
        
        Time Complexity: O(n) where n is total non-zero elements
        """
        if (self.rows, self.cols) != (other.rows, other.cols):
            raise MatrixDimensionError(
                f"Cannot subtract matrices of different dimensions: "
                f"{self.rows}x{self.cols} and {other.rows}x{other.cols}"
            )
        
        result = SparseMatrix(rows=self.rows, cols=self.cols)
        
        # Add elements from self
        for row in self.data:
            for col, value in self.data[row].items():
                result.set_element(row, col, value)
        
        # Subtract elements from other
        for row in other.data:
            for col, value in other.data[row].items():
                current = result.get_element(row, col)
                result.set_element(row, col, current - value)
        
        return result
    
    def multiply(self, other):
        """
        Multiply two sparse matrices using optimized CSR format
        
        Algorithm:
        1. Convert both matrices to CSR format
        2. Use sparse matrix multiplication algorithm optimized for CSR
        3. Convert result back to dictionary format
        
        Time Complexity: O(n*m) where n,m are non-zero elements
        Space Complexity: O(n+m) for CSR storage
        """
        if self.cols != other.rows:
            raise MatrixDimensionError(
                f"Invalid dimensions for multiplication: {self.rows}x{self.cols} and {other.rows}x{other.cols}. "
                f"First matrix columns ({self.cols}) must match second matrix rows ({other.rows})"
            )
        
        result = SparseMatrix(rows=self.rows, cols=other.cols)
        
        # Optimized multiplication directly using dictionary format
        for row in self.data:
            row_data = {}  # Temporary storage for row results
            for k, v1 in self.data[row].items():
                if k in other.data:
                    for col, v2 in other.data[k].items():
                        row_data[col] = row_data.get(col, 0) + v1 * v2
            
            # Only store non-zero results
            for col, value in row_data.items():
                if value != 0:
                    if row not in result.data:
                        result.data[row] = {}
                    result.data[row][col] = value
                    result.nnz += 1
        
        return result
    
    def get_element(self, row, col):
        """
        Get element at specified position
        
        Time Complexity: O(1)
        """
        if row >= self.rows or col >= self.cols:
            return 0  # Return 0 for any position outside current dimensions
        return self.data.get(row, {}).get(col, 0)
    
    def set_element(self, row, col, value):
        """
        Set element at specified position
        
        Time Complexity: O(1)
        """
        # Validate indices
        if row < 0 or row >= self.rows:
            raise MatrixIndexError(f"Row index {row} out of range [0, {self.rows-1}]")
        if col < 0 or col >= self.cols:
            raise MatrixIndexError(f"Column index {col} out of range [0, {self.cols-1}]")
        
        if value != 0:
            if row not in self.data:
                self.data[row] = {}
            self.data[row][col] = value
        elif row in self.data and col in self.data[row]:
            del self.data[row][col]
            if not self.data[row]:
                del self.data[row]
    
    def transpose(self):
        """
        Transpose the matrix
        
        Time Complexity: O(n) where n is number of non-zero elements
        """
        result = SparseMatrix(rows=self.cols, cols=self.rows)
        
        for row in self.data:
            for col, value in self.data[row].items():
                result.set_element(col, row, value)
        
        return result
    
    def save_to_file(self, file_path):
        """Save matrix to file in specified format"""
        try:
            with open(file_path, 'w') as f:
                # Write dimensions in the required format
                f.write(f"rows={self.rows}\n")
                f.write(f"cols={self.cols}\n")
                
                # Write elements with proper spacing after commas
                for row in sorted(self.data.keys()):
                    for col in sorted(self.data[row].keys()):
                        value = self.data[row][col]
                        if value != 0:
                            f.write(f"({row}, {col}, {value})\n")
                            
        except IOError:
            raise IOError(f"Error writing to file: {file_path}")

def ensure_results_directory():
    """Create results directory if it doesn't exist"""
    import os
    results_dir = "../../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def process_matrices(file1_path, file2_path):
    """Process matrices with all operations and save results"""
    try:
        # Create results directory
        results_dir = ensure_results_directory()
        
        # Load matrices
        print(f"\nLoading matrices from:")
        print(f"Matrix 1: {file1_path}")
        print(f"Matrix 2: {file2_path}")
        
        matrix1 = SparseMatrix(file1_path)
        matrix2 = SparseMatrix(file2_path)
        
        print(f"\nMatrix 1 dimensions: {matrix1.rows}x{matrix1.cols} ({matrix1.nnz} non-zero elements)")
        print(f"Matrix 2 dimensions: {matrix2.rows}x{matrix2.cols} ({matrix2.nnz} non-zero elements)")
        
        # Perform multiplication
        try:
            print("\nPerforming multiplication...")
            result = matrix1.multiply(matrix2)
            output_file = f"{results_dir}/multiply_result.txt"
            result.save_to_file(output_file)
            print(f"Saved multiplication result to: {output_file}")
        except Exception as e:
            print(f"Error in multiplication: {str(e)}")
            
        # Perform addition
        try:
            print("\nPerforming addition...")
            # First transpose matrix2 to match matrix1 dimensions if needed
            if matrix1.cols != matrix2.cols or matrix1.rows != matrix2.rows:
                matrix2_transposed = matrix2.transpose()
                result = matrix1.add(matrix2_transposed)
            else:
                result = matrix1.add(matrix2)
            output_file = f"{results_dir}/addition_result.txt"
            result.save_to_file(output_file)
            print(f"Saved addition result to: {output_file}")
        except Exception as e:
            print(f"Error in addition: {str(e)}")
            
        # Perform subtraction
        try:
            print("\nPerforming subtraction...")
            # First transpose matrix2 to match matrix1 dimensions if needed
            if matrix1.cols != matrix2.cols or matrix1.rows != matrix2.rows:
                matrix2_transposed = matrix2.transpose()
                result = matrix1.subtract(matrix2_transposed)
            else:
                result = matrix1.subtract(matrix2)
            output_file = f"{results_dir}/subtraction_result.txt"
            result.save_to_file(output_file)
            print(f"Saved subtraction result to: {output_file}")
        except Exception as e:
            print(f"Error in subtraction: {str(e)}")
        
        print("\nAll operations completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    """Interactive matrix operations with enhanced validation"""
    print("Sparse Matrix Operations")
    print("=======================")
    
    # Get input files
    print("\nEnter the paths to the input files:")
    file1 = input("First matrix file path (or press Enter for default): ").strip()
    file2 = input("Second matrix file path (or press Enter for default): ").strip()
    
    if not file1:
        file1 = "../../sample_inputs/easy_sample_01_2.txt"
    if not file2:
        file2 = "../../sample_inputs/easy_sample_01_3.txt"
    
    while True:
        print("\nAvailable operations:")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. All operations")
        print("5. View matrix information")
        print("6. Exit")
        
        choice = input("\nSelect operation (1-6): ").strip()
        
        if choice == '6':
            print("\nExiting program...")
            break
            
        try:
            results_dir = ensure_results_directory()
            
            # Load matrices
            print(f"\nLoading matrices from:")
            print(f"Matrix 1: {file1}")
            print(f"Matrix 2: {file2}")
            
            matrix1 = SparseMatrix(file1)
            matrix2 = SparseMatrix(file2)
            
            print(f"\nMatrix 1 dimensions: {matrix1.rows}x{matrix1.cols} ({matrix1.nnz} non-zero elements)")
            print(f"Matrix 2 dimensions: {matrix2.rows}x{matrix2.cols} ({matrix2.nnz} non-zero elements)")
            
            if choice == '5':
                continue
            
            if choice in ['1', '4']:  # Addition
                try:
                    print("\nPerforming addition...")
                    # Transpose matrix2 to match matrix1 dimensions
                    matrix2_transposed = matrix2.transpose()
                    print(f"Matrix 2 transposed dimensions: {matrix2_transposed.rows}x{matrix2_transposed.cols}")
                    result = matrix1.add(matrix2_transposed)
                    output_file = f"{results_dir}/addition_result.txt"
                    result.save_to_file(output_file)
                    print(f"Saved addition result to: {output_file}")
                except Exception as e:
                    print(f"Error in addition: {str(e)}")
            
            if choice in ['2', '4']:  # Subtraction
                try:
                    print("\nPerforming subtraction...")
                    # Transpose matrix2 to match matrix1 dimensions
                    matrix2_transposed = matrix2.transpose()
                    print(f"Matrix 2 transposed dimensions: {matrix2_transposed.rows}x{matrix2_transposed.cols}")
                    result = matrix1.subtract(matrix2_transposed)
                    output_file = f"{results_dir}/subtraction_result.txt"
                    result.save_to_file(output_file)
                    print(f"Saved subtraction result to: {output_file}")
                except Exception as e:
                    print(f"Error in subtraction: {str(e)}")
            
            if choice in ['3', '4']:  # Multiplication
                try:
                    print("\nPerforming multiplication...")
                    result = matrix1.multiply(matrix2)
                    output_file = f"{results_dir}/multiply_result.txt"
                    result.save_to_file(output_file)
                    print(f"Saved multiplication result to: {output_file}")
                except Exception as e:
                    print(f"Error in multiplication: {str(e)}")
            
            if choice not in ['1', '2', '3', '4', '5']:
                print("\nInvalid choice! Please select 1-6.")
                continue
                
            print("\nOperations completed!")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 