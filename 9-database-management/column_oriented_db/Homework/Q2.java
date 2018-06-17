import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapred.JobConf;
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

// import java.util.StringTokenizer;
// import java.util.ArrayList;
// import java.util.Arrays;
// import java.util.HashSet;
// import java.util.TreeMap;
// import java.util.Iterator; 
// import java.util.Map.Entry;
import java.util.Map.Entry;
import java.util.*;


public class Q2 {

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

	public static class doc_auths_Reducer extends Reducer <IntWritable, IntWritable, IntWritable, Text> {
		private Text result = new Text();
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
            // convert list to string
            String res = "";
            for (IntWritable s : list){
			    res += s.toString() + ",";
			}
			result.set(res);
			context.write(key, result);
		}
	}

	// ------------ Job 2 -----------------

	public static class auth_coauth_num_Mapper extends Mapper<Object, Text, Text, IntWritable> {
		private Text auth_coauth = new Text();
		private IntWritable one = new IntWritable(1);
		protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String[] tokens = value.toString().split("\t");
			String[] coauths = tokens[1].split(",");
			if (coauths.length>1){
				for (String a: coauths) {
					for (String c: coauths) {
						if (!a.equals("c")) {
							String word = a+"-"+c;
							auth_coauth.set(word); 
							context.write(auth_coauth, one); 
						}
						
					}
					
				}
			}
			
		}
	}

	public static class auth_coauth_num_Reducer extends Reducer <Text, IntWritable, Text, IntWritable> {
		private IntWritable result = new IntWritable();
		public void reduce(Text key, Iterable <IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val: values) {
				sum += val.get();
			}
			result.set(sum);
			context.write(key, result);
		}
	}

	public static class auth_coauths_Mapper extends Mapper<Object, Text, IntWritable, Text> {
		private IntWritable auth = new IntWritable();
		private Text coauth = new Text();
		protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String[] tokens = value.toString().split("\t");
			int n = Integer.parseInt(tokens[1]);
			String[] auths = tokens[0].split("-");

			auth.set(Integer.parseInt(auths[0]));
			String coauth_n = auths[1]+","+tokens[1];
			coauth.set(coauth_n);
			context.write(auth, coauth);

			coauth_n = "";
			auth.set(Integer.parseInt(auths[1]));
			coauth_n = auths[0]+","+tokens[1];
			coauth.set(coauth_n);
			context.write(auth, coauth);
		}
	}

	public static class auth_coauths_Reducer extends Reducer <IntWritable, Text, IntWritable, Text> {
		private Text result = new Text();
		public void reduce(IntWritable key, Iterable <Text> values, Context context) throws IOException, InterruptedException {
			TreeMap<Integer,Integer> tm = new TreeMap<Integer,Integer>(new Comparator<Integer>(){
                        @Override
                        public int compare(Integer o1, Integer o2) {
                            if(o1>o2){
					            return 1; 
					        } else {
					            return -1;
					        } 
                        }
                    });
			// Sort coauth in a treemap
			for (Text val: values) {
				String[] coauth_n = val.toString().split(",");
				int coauth = Integer.parseInt(coauth_n[0]);
				int n = Integer.parseInt(coauth_n[1]);
				tm.put(n,coauth);
				// context.write(key, val);
			}
			// Select the top k
			Configuration conf = context.getConfiguration();
			int k = conf.getInt("top_k2",3);

			Iterator<Entry<Integer , Integer>> it = tm.entrySet().iterator();
			Entry<Integer , Integer> entry2 = null;
			HashSet<Integer> keys = new HashSet<Integer> ();
			keys.addAll(tm.keySet());
			while(keys.size()>k){
				entry2 = it.next();
				it.remove();
				keys.clear();
				keys.addAll(tm.keySet());
			}
			// output 
			String out = "";
			for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
				out += "<"+ entry.getValue() +", "+ entry.getKey()+">";
				out += "\t"; 
			}
			result.set(out);
			context.write(key, result);
		}
	}	

	

	public static void main(String[] args) throws Exception {
		Path TEMP_PATH = new Path("temp1");
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "job1");
		job.setJarByClass(Q2.class);
		job.setMapperClass(doc_auth_Mapper.class);
		job.setReducerClass(doc_auths_Reducer.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, TEMP_PATH);
		job.waitForCompletion(true);
		// Chain using others jobs
		Configuration conf2 = new Configuration();
		Job job2 = Job.getInstance(conf2, "job2");
		job2.setJarByClass(Q2.class);
		job2.setMapperClass(auth_coauth_num_Mapper.class);
		job2.setReducerClass(auth_coauth_num_Reducer.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(IntWritable.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(IntWritable.class);
		// job2.setInputFormatClass(SequenceFileInputFormat.class);
		FileInputFormat.addInputPath(job2, TEMP_PATH);
		Path TEMP_PATH2 = new Path("temp2");
		FileOutputFormat.setOutputPath(job2, TEMP_PATH2);
		job2.waitForCompletion(true);

		// Job3
		// Configuration conf3 = new Configuration();
		JobConf conf3 = new JobConf();
		Job job3 = Job.getInstance(conf3, "job3");
		System.out.println("**********************************************************  "+args[2]);
		String s = args[2];
		conf3.setInt("top_k", Integer.parseInt(s));
		conf3.setInt("top_k2", Integer.parseInt(s));
		System.out.println("**********************************************************  "+args[2]);
		conf3.set("mapred.reduce.tasks", "1");
		job3.setNumReduceTasks(1);
		job3.setJarByClass(Q2.class);
		job3.setMapperClass(auth_coauths_Mapper.class);
		job3.setReducerClass(auth_coauths_Reducer.class);
		job3.setMapOutputKeyClass(IntWritable.class);
		job3.setMapOutputValueClass(Text.class);
		job3.setOutputKeyClass(IntWritable.class);
		job3.setOutputValueClass(Text.class);
		// job2.setInputFormatClass(SequenceFileInputFormat.class);
		FileInputFormat.addInputPath(job3, TEMP_PATH2);
		FileOutputFormat.setOutputPath(job3, new Path(args[1]));
		boolean end = job3.waitForCompletion(true);
		TEMP_PATH.getFileSystem(conf).delete(TEMP_PATH);
		TEMP_PATH2.getFileSystem(conf).delete(TEMP_PATH2);

		System.exit(end ? 0 : 1);

	}
}