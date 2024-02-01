import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.function.Function;

import de.javagl.mnist.reader.MnistEntry;

public class Networkv2 {

	Matrix[] activations;
	Matrix[] zVals;
	Matrix[] weights;
	Matrix[] biases;
	
	
	//Sigmoid Functions
	Function<Double, Double> sigmoid 			= (x) -> (1/( 1 + Math.pow(Math.E,(-1*x))));
	Function<Double, Double> sigmoid1Derivative = (x) -> sigmoid.apply(x)*(1-sigmoid.apply(x));
	

	public Networkv2(int[] s) {
		
		//Set activations
		activations = new Matrix[s.length];
		zVals       = new Matrix[s.length];
		
		
		for (int i = 0; i < s.length; i++) { 
			activations[i] = new Matrix(s[i], 1);
			zVals[i]       = new Matrix(s[i], 1);
		}
		
		
		//Set weights
		weights = new Matrix[s.length-1];
		biases  = new Matrix[s.length-1];
		
		for (int i = 1; i < s.length; i ++) {
			
			weights[i-1] = new Matrix(s[i], s[i-1]);
			biases [i-1] = new Matrix(s[i], 1); 
			
			weights[i-1].randomize();
			biases [i-1].randomize();
			
		}
		
	}	
	
	public void feedForward(Matrix inputs) throws CloneNotSupportedException {
		
		activations[0] = inputs;
		
		for (int i = 0; i < activations.length-1; i++) {
			
			activations[i+1] = Matrix.matrixProduct(weights[i], activations[i]);
			activations[i+1].matrixAdd(biases[i]);
			zVals[i] = (Matrix) activations[i+1].clone();
			activations[i+1].mapFunction(sigmoid);
			
		}
	}
	
	public void trainBatch(MnistEntry[] mE) throws CloneNotSupportedException {
		
		int batchSize = mE.length;
		
		//Deltas
		Matrix[] deltas = new Matrix[biases.length];
		
		//Averaged partial derivatives
		Matrix[] partDerivWeights = new Matrix[weights.length];
		Matrix[] partDerivBiases  = new Matrix[biases .length];
		
		for (int i = 0; i < mE.length; i++) {
			
			feedForward(new Matrix ( mE[i].getFormattedData() ) ); 
			
			//Calculate deltas for last layer
			
			Matrix tmpMatrix = (Matrix) activations[activations.length-1].clone();
				
			// (aL - y)
			double tmp = tmpMatrix.vals[mE[i].getLabel()][0] - 1.0;

			tmpMatrix.setVal(mE[i].getLabel(), 0, tmp);

			zVals[zVals.length-1].mapFunction(sigmoid1Derivative);
			
			deltas[deltas.length-1] = Matrix.hadamardProduct(tmpMatrix, zVals[zVals.length-1]);
			
			
			//Calculate other deltas
			for (int j = deltas.length-2; j >= 0; j--) {
				
				//System.out.println(j);
				
				tmpMatrix = Matrix.matrixProduct( Matrix.transpose(weights[j+1]), deltas[j+1] );
				
				zVals[j+1].mapFunction(sigmoid1Derivative);
				
				//System.out.println(zVals[j].rows);
				//System.out.println(tmpMatrix.rows);
				
				deltas[j] = Matrix.hadamardProduct(tmpMatrix, zVals[j]);
				
			}
			
			//Calculate gradients
			for (int n = 0; n < deltas.length; n++) {
				
				partDerivBiases[n] = new Matrix(biases[n].rows, 1);
				
				//System.out.println(deltas[n]);
				
				partDerivBiases[n].matrixAdd(deltas[n]);
				
				partDerivWeights[n] = new Matrix(weights[n].rows, weights[n].cols);
				
				for (int j = 0; j < weights[n].rows; j++) {
					for (int k = 0; k < weights[n].cols; k++) {
						
						//System.out.println(j + "==" + k);
						
						partDerivWeights[n].vals[j][k] = partDerivWeights[n].vals[j][k] + ( activations[n].vals[k][0] * deltas[n].vals[j][0] );
						
					}
				}
				
			}

		}
		
		//Average and get negative gradient
		Function<Double, Double> divideBy = (x) -> -12*(x / batchSize);
		
		for (int i = 0; i < partDerivBiases.length; i++) {
			
			partDerivBiases [i].mapFunction(divideBy);
			partDerivWeights[i].mapFunction(divideBy);
			
		}
		
		for (int i = 0; i < partDerivBiases.length; i++) {
			
			biases [i].matrixAdd(partDerivBiases [i]);
			weights[i].matrixAdd(partDerivWeights[i]);
			
		}
	}
	
	
	
	
	
	public void testAccuracy(ArrayList<MnistEntry> testingData) throws CloneNotSupportedException {
		
		int n = testingData.size();
		
		int nCorrect = 0;
		
		int[] issues = new int[10];
		
		
		for (int i = 0; i < n; i++) {
			
			feedForward( new Matrix( testingData.get(i).getFormattedData() ) );
			
			double tmp = 0;
			int index = 0;
			
			for (int j = 0; j < 10; j++) {
				
				double res = activations[activations.length-1].vals[j][0];
				if (res > tmp) {
					tmp = res;
					index = j;
				}
				
			}
			
			if (index == testingData.get(i).getLabel()) nCorrect++;
			else issues[testingData.get(i).getLabel()] = issues[testingData.get(i).getLabel()] + 1;
			
		}
		
		System.out.println(n + "/" + nCorrect);
		System.out.println((nCorrect/100.0) + "% Accuracy");
		for (int i = 0; i < issues.length; i++) System.out.print(i + ": " + issues[i] + "	");
		System.out.println();
		
		
		
		
	}
	
	

	
	
	
	
	
	

}
