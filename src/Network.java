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

public class Network {
	
	int[] size;
	
	Matrix[] weights;
	Matrix[] biases;
	
	//Sigmoid Functions
	Function<Double, Double> sigmoid 			= (x) -> (1/( 1 + Math.pow(Math.E,(-1*x))));
	Function<Double, Double> sigmoid1Derivative = (x) -> sigmoid.apply(x)*(1-sigmoid.apply(x));
	
	public Network(int[] s) {
		
		weights = new Matrix[s.length-1];
		biases  = new Matrix[s.length-1];
		
		size = s;
		
		for (int i = 1; i < s.length; i ++) {
			weights[i-1] = new Matrix(s[i], s[i-1]);
			biases [i-1] = new Matrix(s[i], 1); 
			
			weights[i-1].randomize();
			biases [i-1].randomize();
		}
	}
	
	public Matrix[] feedForward(MnistEntry m) {
		
		Matrix[] activations = new Matrix[size.length];
		activations[0] = new Matrix(m.getFormattedData());
		for (int i = 1; i < size.length; i++) {
			
			activations[i] = Matrix.matrixProduct(weights[i-1], activations[i-1]);
			activations[i].  matrixAdd(biases[i-1]);
			activations[i].  mapFunction(sigmoid);
			
		}
		return activations;
	}
	
	
	public Matrix[][] feedForward2(MnistEntry m) {
		
		//The activations of the output layer will be added at the end of this array
		//This is helpful when calculating deltas
		Matrix[] activations = new Matrix[size.length];
		Matrix[] zVals       = new Matrix[size.length-1];

		activations[0] = new Matrix(m.getFormattedData());
		
		for (int i = 1; i < size.length; i++) {
			
			activations[i] = Matrix.matrixProduct(weights[i-1], activations[i-1]);
			activations[i].  matrixAdd(biases[i-1]);
			
			zVals[i-1] = new Matrix(activations[i]);
			
			activations[i].  mapFunction(sigmoid);
			
		}
		
		return new Matrix[][] {activations, zVals};
	}
	
	
	
	
	
	public Matrix[][] calcIndividualGradient(MnistEntry m) {
		
		Matrix[] deltas = new Matrix[size.length-1];
		
		Matrix[][] output = feedForward2(m);
		
		Matrix[] activations = output[0]; 
		Matrix[] zValues	 = output[1];

		
		//Run zVals through 1st derivative of sigmoid function
		//these are the amount of layers - 1 of indices at the end of the array
		for (int i = 0; i < zValues.length; i++) {
			zValues[i].mapFunction(sigmoid1Derivative);
		}
		
		//Desired output
		double[] yArray = new double[size[size.length-1]];
		yArray[m.getLabel()] = -1.0;
		Matrix costDeriv = new Matrix(yArray);
		
		//Calc (a[L] - y)
		costDeriv.matrixAdd(activations[size.length-1]);
		
		//Deltas of output layer
		deltas[deltas.length-1] = Matrix.hadamardProduct(costDeriv, zValues[zValues.length-1]);
		
		//Deltas of other layers
		for (int i = deltas.length-2; i >= 0; i--) {
			
			Matrix transposedWeights = Matrix.transpose(weights[i+1]);
			//  ( ( w[l+1] )T * delta[l+1] )
			Matrix z = Matrix.matrixProduct(transposedWeights, deltas[i+1]);
			Matrix delta = Matrix.hadamardProduct(z, zValues[i]);
			
			deltas[i] = delta;
		}
		
		//Calculate gradients
		//Bias gradients are the deltas
		Matrix[] weightGradients = new Matrix[size.length-1];
		
		//Set dimensions to be correct
		for (int i = 0; i < weightGradients.length; i++) weightGradients[i] = new Matrix(weights[i]);
		
		//Iterate through weights
		for (int i = 0; i < weightGradients.length; i++) {
			
			for (int j = 0; j < weightGradients[i].rows; j++) {
				
				for (int k = 0; k < weightGradients[i].cols; k++) {
					
					weightGradients[i].setVal(j, k, activations[i].vals[k][0]*deltas[i].vals[j][0]);

				}
				
			}

		}
		

		return new Matrix[][] {weightGradients, deltas};
	}
	
	
	public void trainNetwork(ArrayList<MnistEntry> trainingData) {
		
		ArrayList<Matrix[]> weightGradients = new ArrayList<Matrix[]>();
		ArrayList<Matrix[]> biasGradients   = new ArrayList<Matrix[]>();
		
		System.out.println("|---------------Training--Progress---------------|");
						
		
		for (int i = 0; i < trainingData.size(); i++) {
			
			Matrix[][] output = calcIndividualGradient(trainingData.get(i));
			
			weightGradients.add(output[0]);
			biasGradients.  add(output[1]);
			
			if ( (i+1) % 100 == 0 ) {
				
				Matrix[][] allWG = new Matrix[size.length-1][100];
				Matrix[][] allBG = new Matrix[size.length-1][100];
				
				if ((i+1) % 1200 == 0) System.out.print("#");
				
				
				for (int l = 0; l < size.length-1; l++) {
					for (int f = 0; f < 100; f++) {	
						allWG[l][f] = weightGradients.get(f)[l];
						allBG[l][f] = biasGradients  .get(f)[l];
					}
				}
				
				Matrix[] averageWG = new Matrix[size.length-1];
				Matrix[] averageBG = new Matrix[size.length-1];

				
				for (int k = 0; k < averageWG.length; k++) averageWG[k] = Matrix.averageMatricies(allWG[k]);
				for (int k = 0; k < averageBG.length; k++) averageBG[k] = Matrix.averageMatricies(allBG[k]);
				
				
				//Apply gradient
				
				//turn to negative gradient
				
				Function<Double, Double> negative = (x) -> -8*x;
						
				for (int k = 0; k < averageWG.length; k++) averageWG[k].mapFunction(negative);
				for (int k = 0; k < averageBG.length; k++) averageBG[k].mapFunction(negative);
				
				//Add gradients to weights and biases
				for (int k = 0; k < averageWG.length; k++) weights[k].matrixAdd(averageWG[k]);
				for (int k = 0; k < averageWG.length; k++) biases [k].matrixAdd(averageBG[k]);
				
				//Reset ArrayLists
				weightGradients.clear();
				biasGradients.  clear();

			}
			
			
			
		}
		
		System.out.println();
		
	}
	
	
	
	public void testAccuracy(ArrayList<MnistEntry> testingData) {
		
		int n = testingData.size();
		
		int nCorrect = 0;
		
		int[] issues = new int[10];
		
		
		for (int i = 0; i < n; i++) {
			
			Matrix[] output = feedForward(testingData.get(i));
			
			double tmp = 0;
			int index = 0;
			
			for (int j = 0; j < 10; j++) {
				
				double res = output[size.length-1].vals[j][0];
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
	
	
	public void testImage(double[] input, int label) {
		
		Matrix[] activations = new Matrix[size.length];
		activations[0] = new Matrix(input);
		for (int i = 1; i < size.length; i++) {
			
			activations[i] = Matrix.matrixProduct(weights[i-1], activations[i-1]);
			activations[i].  matrixAdd(biases[i-1]);
			activations[i].  mapFunction(sigmoid);
			
		}
		
		double tmp = 0;
		int index = 0;
		
		System.out.println("=======================================================================================");
		
		for (int j = 0; j < 10; j++) {
			
			System.out.println(j + ":	" + (int)((activations[size.length-1].vals[j][0]+0.005)*10000)/100 + "%");
			
			double res = activations[size.length-1].vals[j][0];
			if (res > tmp) {
				tmp = res;
				index = j;
			}
		}
		
		System.out.println("=======================================================================================");
		
		System.out.println("Image Number: " + label);
		
		if (index == label) {
			System.out.println("Correct");
		}
		
	}
	
	public void saveNetworkToFile() throws FileNotFoundException, UnsupportedEncodingException {
		//Weights
		PrintWriter writer = new PrintWriter("./prevNetwork/weights.txt", "UTF-8");
		
		for (int i = 0; i < weights.length; i++) {
			
			//writer.println(weights[i].rows);
			//writer.println(weights[i].cols);
			
			for (int m = 0; m < weights[i].rows; m++) {
				
				for (int n = 0; n < weights[i].cols; n++) {
					
					writer.println(weights[i].vals[m][n]);
					
				}
			}
		}
		
		writer.close();
		
		//Biases
		writer = new PrintWriter("./prevNetwork/biases.txt", "UTF-8");
		
		for (int i = 0; i < biases.length; i++) {
		
			//writer.println(biases[i].rows);
			
			for (int m = 0; m < biases[i].rows; m++) {

				writer.println(biases[i].vals[m][0]);
			}
		}
		writer.close();
		
	}
	
	public void readFile() throws IOException {
		
		BufferedReader br = Files.newBufferedReader(Paths.get("./prevNetwork/weights.txt"));
		
		for (int i = 0; i < weights.length; i++) {
			
			for (int m = 0; m < weights[i].rows; m++) {
				
				for (int n = 0; n < weights[i].cols; n++) {
					
					weights[i].vals[m][n] = Double.valueOf(br.readLine());
					
				}
			}
		
		}
		
		br.close();
		
		br = Files.newBufferedReader(Paths.get("./prevNetwork/biases.txt"));
        	
		for (int i = 0; i < biases.length; i++) {
			
			for (int m = 0; m < biases[i].rows; m++) {

				biases[i].vals[m][0] = Double.valueOf(br.readLine());
			}
		}
		
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

}
