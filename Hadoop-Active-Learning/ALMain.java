

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class ALMain {

	private String select_num;
	private String class_num;
	private int iterationNum;
	private String sourcePath_Test;
	private String outputPath;
	private Configuration conf;

	public ALMain(String select_num,String class_num,int iterationNum, String sourcePath_Test, String outputPath,
			Configuration conf) {
		this.select_num = select_num;
		this.class_num = class_num;
		this.iterationNum = iterationNum;
		this.sourcePath_Test = sourcePath_Test;
		this.outputPath = outputPath;
		this.conf = conf;
	}

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		System.out.println("-----start-----");
		long starTime = System.currentTimeMillis();
		Configuration conf = new Configuration();
		
//		10 3 5 file:///C:/Users/new/Desktop/MapReduceAL/test file:///C:/Users/new/Desktop/MapReduceAL/train
		String select_num = args[0];   //选择样例个数10	
		String class_num = args[1];		 //类别个数 3 
		int iterationNum = Integer.parseInt(args[2]);		//迭代次数5
		String sourcePath_Test = args[3];			//测试集路径
		String outputPath = args[4];   				//与训练集路径相同 
		ALMain al = new ALMain(select_num,class_num, iterationNum, sourcePath_Test, outputPath, conf);

		al.ALDriverJob();

	}

	public void ALDriverJob() throws IOException, ClassNotFoundException, InterruptedException {
		long start = System.currentTimeMillis();
		for (int num = 0; num < iterationNum; num++) {
			System.out.println("----------------第"+num+"次迭代-------------------");
			Job ALDriverJob = new Job();
			
			ALDriverJob.setJobName("ALDriverJob"+num);
			ALDriverJob.setJarByClass(AL.class);
			ALDriverJob.getConfiguration().set("trainPath", outputPath+"/train"+num+"/");
			ALDriverJob.getConfiguration().set("select_num", select_num);
			ALDriverJob.getConfiguration().set("class_num", class_num);

			ALDriverJob.setMapperClass(AL.ALMapper.class);
			ALDriverJob.setMapOutputKeyClass(DoubleWritable.class);
			ALDriverJob.setMapOutputValueClass(Text.class);

			ALDriverJob.setReducerClass(AL.ALReduce.class);
			ALDriverJob.setOutputKeyClass(NullWritable.class);
			ALDriverJob.setOutputValueClass(Text.class);

			FileInputFormat.addInputPath(ALDriverJob, new Path(sourcePath_Test));
			FileOutputFormat.setOutputPath(ALDriverJob, new Path(outputPath+"/train"+(num+1)+"/"));

			ALDriverJob.waitForCompletion(true);
		}
		long end = System.currentTimeMillis();
		System.out.println((float)(Math.round(end-start))/1000);
		System.out.println("finished!");
			
	}

}
