import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

public class Q1 {

	static class MyArrayWritable extends ArrayWritable{
		public MyArrayWritable(Writable[] values) {
	        super(IntWritable.class, values);
	    }

	    public MyArrayWritable() {
	        super(IntWritable.class);
	    }
		@Override
		public IntWritable[] get() {
			return (IntWritable[]) super.get();
		}
		@Override
		public String toString() {
			return Arrays.toString(get());
		}
	}

	public static class doc_auth_Mapper extends Mapper<Object, Text, IntWritable, IntWritable> {
		private IntWritable doc = new IntWritable();
		private IntWritable author = new IntWritable();
		protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String[] tokens = value.toString().split("\t");
			// doc.set(Integer.parseInt(tokens[0]));
			// author.set(Integer.parseInt(tokens[1]));
			// context.write(doc, author);
			if (tokens.length == 2) {
				doc.set(Integer.parseInt(tokens[0])); 
				author.set(Integer.parseInt(tokens[1]));
				context.write(doc, author); 
			}
		}
	}

	public static class doc_auths_Reducer extends Reducer <IntWritable, IntWritable, IntWritable, MyArrayWritable> {
		private MyArrayWritable result;
		ArrayList<IntWritable> list;
		public void reduce(IntWritable key, Iterable <IntWritable> values, Context context) throws IOException, InterruptedException {
			list = new ArrayList<IntWritable>();
			for (IntWritable val: values) {
				list.add(new IntWritable(val.get()));
				// list.add(new IntWritable(5));
			}
			// Get unique elements
			HashSet<IntWritable> hs = new HashSet<IntWritable> ();
            hs.addAll(list);
            list.clear();
            list.addAll(hs);
			result = new MyArrayWritable(list.toArray(new IntWritable[list.size()]));
			context.write(key, result);
		}
	}

	public static class auth_coauth_Mapper extends Mapper<IntWritable, MyArrayWritable, IntWritable, IntWritable> {
		private IntWritable doc = new IntWritable();
		private IntWritable author = new IntWritable();
		protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String[] tokens = value.toString().split("\t");
			// doc.set(Integer.parseInt(tokens[0]));
			// author.set(Integer.parseInt(tokens[1]));
			// context.write(doc, author);
			if (tokens.length == 2) {
				doc.set(Integer.parseInt(tokens[0])); 
				author.set(Integer.parseInt(tokens[1]));
				context.write(doc, author); 
			}
		}
	}

	// public static class doc_auths_Reducer extends Reducer <IntWritable, IntWritable, IntWritable, ArrayWritable> {
	// 	// private MyArrayWritable result;
	// 	private ArrayWritable result = new ArrayWritable(IntWritable.class);
		
	// 	public void reduce(IntWritable key, Iterable <IntWritable> values, Context context) throws IOException, InterruptedException {
	// 		ArrayList<IntWritable> list; list = new ArrayList<IntWritable>();
	// 		for (IntWritable val: values) {
	// 			list.add(val);
	// 		}
	// 		// result = new ArrayWritable(IntWritable.class, list.toArray(new IntWritable[list.size()]));
	// 		result.set(list.toArray(new IntWritable[list.size()]));
	// 		context.write(key, result);
	// 	}
	// }



	// public static class doc_auths_Reducer extends Reducer <IntWritable, IntWritable, IntWritable, ArrayWritable> {
	// 	// private ArrayWritable result = new ArrayWritable();
	// 	ArrayList<String> list;
	// 	String [] list2;
	// 	private ArrayWritable result;
	// 	String[] errorSoon = {"Hello", "World"};
	// 	public void reduce(IntWritable key, Iterable <IntWritable> values, Context context) throws IOException, InterruptedException {
	// 		list = new ArrayList<String>();
	// 		for (IntWritable val: values) {
	// 			list.add(val.toString());
	// 		}
	// 		list2 = list.toArray(new String[list.size()]);
	// 		System.out.println(list2);
	// 		result = new ArrayWritable(errorSoon);
	// 		context.write(key, result);
	// 	}
	// }

	// public static class doc_auths_Reducer extends Reducer <IntWritable, IntWritable, IntWritable, Text> {
	// 	// private IntWritable[] result = new IntWritable();
	// 	private Text authors = new Text();
	// 	public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
	// 	throws IOException, InterruptedException {
	// 		String res = "";
	// 		for (IntWritable val: values) {
	// 			res += "#";
	// 			// res += IntWritable.toString(val);
	// 			res += val.toString();
	// 		}
	// 		authors.set(res);
	// 		context.write(key, authors);
	// 	}
	// }

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "Q1");
		job.setJarByClass(Q1.class);
		job.setMapperClass(doc_auth_Mapper.class);
		// job.setCombinerClass(doc_auths_Reducer.class);
		job.setReducerClass(doc_auths_Reducer.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(MyArrayWritable.class);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		// Chain using others jobs
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}