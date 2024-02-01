import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class ReadImage {
	
	
	
	public void testImage(String fileName, Network n) throws IOException {
		
		File imgPath = new File("./images/" + fileName);
		
		int iEnd = fileName.indexOf("."); 
		String labelString = "999";
		if (iEnd != -1) labelString = fileName.substring(0 , iEnd); 
		
		int label = Integer.parseInt(labelString);			
		
		double[] data = getImageData(imgPath);
		
		n.testImage(data, label);
		
		
		
	}
	
	public double[] getImageData(File f) throws IOException {
		
		BufferedImage bi = ImageIO.read(f);
		
		double[] vals = new double[784];
		
		for (int row = 0; row < 28; row++) {
			
			for (int col = 0; col < 28; col++) {
				
				int rgb = bi.getRGB(col, row);
				
				int red   = (rgb >> 16) & 0xFF;
				int green = (rgb >>  8) & 0xFF;
				int blue  = (rgb      ) & 0xFF;
				
				double val = (((red+green+blue) / 765.0)-1)*-1;
				
				if (val >= 0.5) {
					val = 1;
				} else {
					val = 0;
				}
				
				vals[row*28+col] = val;
				
				
				
				//System.out.print(val + " ");
				
			}
			
			//System.out.println();
		}
		
		
		return vals;
		
	}

}
