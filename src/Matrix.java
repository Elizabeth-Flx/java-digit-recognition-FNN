import java.util.Random;
import java.util.function.Function;

public class Matrix implements Cloneable {
	
	Random rand = new Random();
	
	int rows;
	int cols;
	
	double[][] vals;
	
	//====================
	//Constructor
	public Matrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		vals = new double[rows][cols];
	}
	
	public Matrix(double[] data) {
		rows = data.length;
		cols = 1;
		vals = new double[rows][cols];
		for (int i = 0; i < rows; i++) vals[i][0] = data[i];
	}
	

	//====================
    //--Set Specific Value
	public void setVal(int row, int col, double val) {
		vals[row][col] = val;
	}
	
	
	
	//====================
	//--Randomize Matrix
	public void randomize() {
		for (int i = 0; i < vals.length; i++) {
			for (int j = 0; j < vals[i].length; j++) {
				vals[i][j] = rand.nextGaussian();
			}
		}
	}
	
	//====================
	//--Apply Function
	public void mapFunction(Function<Double, Double> fn) {
		for (int i = 0; i < vals.length; i++) {
			for (int j = 0; j < vals[i].length; j++) {
				vals[i][j] = fn.apply(vals[i][j]);
			}
		}
	}
	
	//====================
	//--Matrix Addition
	public void matrixAdd(Matrix b) {
		if (rows != b.rows || cols != b.cols) {
			System.out.println("Matrix sizes don't match.");
		}
		for (int i = 0; i < vals.length; i++) {
			for (int j = 0; j < vals[i].length; j++) {
				vals[i][j] += b.vals[i][j];
			}
		}
	}
	
	
	
	//====================
	//Print Matrix
	public void printMatrix() {
		System.out.println("=======================================================================================");
		for (int i = 0; i < vals.length; i++) {
			for (int j = 0; j < vals[i].length; j++) {
				System.out.print(vals[i][j] + "	");
			}
			System.out.println();
		}
		System.out.println("=======================================================================================");
	}

		
	//==================================|
	//Static Methods					|
	//==================================|
	
	//=========================
	//--Scalar Multiplication
	public static Matrix scalarMult(Matrix a, double b) {
		Matrix result = new Matrix(a.rows, a.cols);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.cols; j++) {
				result.vals[i][j] = a.vals[i][j] * b;
			}
		}
		return result;
	}
	

	//=========================
	//--Hadamard Product
	public static Matrix hadamardProduct(Matrix a, Matrix b) {
		if (a.rows != b.rows || a.cols != b.cols) {
			System.out.println("Matrix sizes don't match.");
			return null;
		}
		Matrix result = new Matrix(a.rows, a.cols);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.cols; j++) {
				result.vals[i][j] = a.vals[i][j] * b.vals[i][j];
			}
		}
		return result;
	}
	
	
	//=========================
	//--Matrix Product
	public static Matrix matrixProduct(Matrix a, Matrix b) {
		if (a.cols != b.rows) {
			System.out.println("Matrix sizes don't match");
			System.out.println("a.cols: " + a.cols + " != " + "b.rows: " + b.rows);
			
			return null;
		}
		Matrix result = new Matrix(a.rows, b.cols);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < b.cols; j++) {
				double num = 0;
				for (int x = 0; x < a.cols; x++) {
					num += a.vals[i][x] * b.vals[x][j];
				}
				result.vals[i][j] = num;
			}
		}
		return result;
	}


	//=========================
	//--Transpose
	public static Matrix transpose(Matrix m) {
		Matrix result = new Matrix(m.cols, m.rows);
		for (int j = 0; j < m.cols; j++) {
			for (int i = 0; i < m.rows; i++) {
				result.vals[j][i] = m.vals[i][j];
			}
		}
		return result;
	}
	
	
	//=========================
	//--Average Batch
	public static Matrix averageMatricies(Matrix[] batch) {

		Matrix result = new Matrix(batch[0].rows, batch[0].cols);
		for (Matrix m : batch) result.matrixAdd(m);
		
		Function<Double, Double> divideBy = (x) -> x / batch.length;
		result.mapFunction(divideBy);
		
		return result;
		
	}
	
	
	
	
	
	@Override
	public Object clone() throws CloneNotSupportedException {
		return super.clone();
	}
	
	
	
	
}