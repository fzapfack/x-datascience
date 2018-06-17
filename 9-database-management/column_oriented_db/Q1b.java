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
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
// import org.apache.hadoop.mapred.WholeFileInputFormat;
// import org.apache.hadoop.mapred.KeyValueTextInputFormat;

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
		private IntWritable author = new IntWritable();
		private IntWritable coauth = new IntWritable();
		protected void map(IntWritable key, MyArrayWritable value, Context context) throws IOException, InterruptedException {
			IntWritable[] values = value.get();
			for (IntWritable val: values) {
				author = key; 
				coauth = val;
				context.write(author, coauth); 
			}
		}
	}

	public static class auth_coauth_Reducer extends Reducer <IntWritable, IntWritable, IntWritable, IntWritable> {
		private IntWritable result = new IntWritable();
		ArrayList<IntWritable> list;
		public void reduce(IntWritable key, Iterable <IntWritable> values, Context context) throws IOException, InterruptedException {
			list = new ArrayList<IntWritable>();
			for (IntWritable val: values) {
				list.add(new IntWritable(val.get()));
			}
			// Get unique elements
			HashSet<IntWritable> hs = new HashSet<IntWritable> ();
            hs.addAll(list);
            list.clear();
            list.addAll(hs);
            //set result
			result.set(list.size());
			context.write(key, result);
		}
	}


	public static void main(String[] args) throws Exception {
		Path TEMP_PATH = new Path("/user/fzapfack/temp");
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "job1");
		job.setJarByClass(Q1.class);
		job.setMapperClass(doc_auth_Mapper.class);
		// job.setCombinerClass(doc_auths_Reducer.class);
		job.setReducerClass(doc_auths_Reducer.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(MyArrayWritable.class);
		// job1.setInputFormatClass(WholeFileInputFormat.class);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		// FileOutputFormat.setOutputPath(job, new Path(args[1]));
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		SequenceFileOutputFormat.setOutputPath(job, TEMP_PATH);
		job.waitForCompletion(true);
		// Chain using others jobs
		Configuration conf2 = new Configuration();
		Job job2 = Job.getInstance(conf2, "job2");
		job2.setJarByClass(Q1.class);
		job2.setMapperClass(auth_coauth_Mapper.class);
		job2.setReducerClass(auth_coauth_Reducer.class);
		job2.setMapOutputKeyClass(IntWritable.class);
		job2.setMapOutputValueClass(IntWritable.class);
		job2.setOutputKeyClass(IntWritable.class);
		job2.setOutputValueClass(IntWritable.class);
		job2.setInputFormatClass(SequenceFileInputFormat.class);
		SequenceFileInputFormat.addInputPath(job2, TEMP_PATH);
		FileOutputFormat.setOutputPath(job2, new Path(args[1]));
		System.exit(job2.waitForCompletion(true) ? 0 : 1);
	}
}