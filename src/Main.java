import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.function.Consumer;

import de.javagl.mnist.reader.MnistDecompressedReader;
import de.javagl.mnist.reader.MnistEntry;

public class Main {
	
	static Networkv2 network = new Networkv2(new int[] {784, 16, 16, 10});
	
	static ArrayList<MnistEntry> trainingArray = new ArrayList<MnistEntry>();
	static ArrayList<MnistEntry> testingArray  = new ArrayList<MnistEntry>();
	
	static MnistDecompressedReader mnistReader = new MnistDecompressedReader();
		
	static Path inputDirectoryPath = Paths.get("./data");
		
	public static void getData() throws IOException {
			
		Consumer<MnistEntry> trainingCons = (x) -> trainingArray.add(x);
		Consumer<MnistEntry> testingCons  = (x) -> testingArray .add(x);
		
		mnistReader.readDecompressedTraining(inputDirectoryPath, trainingCons);
		mnistReader.readDecompressedTesting (inputDirectoryPath, testingCons );
			
	}
	
	public static void main(String[] args) throws IOException, CloneNotSupportedException {
		
		getData();
		
		
		for (int i = 0; i < 10; i++) {
			System.out.println("<=====< " + (i+1) + ". Epoch >=====>");
			
			MnistEntry[] batch = new MnistEntry[1000];
			int count = 0;
			
			Collections.shuffle(trainingArray);
			
			for (MnistEntry m : trainingArray) {
				
				batch[count] = m;
				count++;
				
				if (count == 1000) {
					count = 0;
					network.trainBatch(batch);
				}
			}
			network.testAccuracy(testingArray);
		}

		
	}
	
	

}
