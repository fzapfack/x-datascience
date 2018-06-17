import java.io.IOException;

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

// import java.util.StringTokenizer;
// import java.util.ArrayList;
// import java.util.Arrays;
// import java.util.HashSet;
// import java.util.TreeMap;
// import java.util.Iterator; 
// import java.util.Map.Entry;
import java.util.Map.Entry;
import java.util.*;


public class Q1 {

	// public static class doc_auth_Mapper extends Mapper<Object, Text, IntWritable, IntWritable> {
	// 	private IntWritable doc = new IntWritable();
	// 	private IntWritable author = new IntWritable();
	// 	protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
	// 		String[] tokens = value.toString().split("\t");
	// 		// doc.set(Integer.parseInt(tokens[0]));
	// 		// author.set(Integer.parseInt(tokens[1]));
	// 		// context.write(doc, author);
	// 		if (tokens.length == 2) {
	// 			doc.set(Integer.parseInt(tokens[0])); 
	// 			author.set(Integer.parseInt(tokens[1]));
	// 			context.write(doc, author); 
	// 		}
	// 	}
	// }

	// public static class doc_auths_Reducer extends Reducer <IntWritable, IntWritable, IntWritable, Text> {
	// 	private Text result = new Text();
	// 	ArrayList<IntWritable> list;
	// 	public void reduce(IntWritable key, Iterable <IntWritable> values, Context context) throws IOException, InterruptedException {
	// 		list = new ArrayList<IntWritable>();
	// 		for (IntWritable val: values) {
	// 			list.add(new IntWritable(val.get()));
	// 			// list.add(new IntWritable(5));
	// 		}
	// 		// Get unique elements
	// 		HashSet<IntWritable> hs = new HashSet<IntWritable> ();
 //            hs.addAll(list);
 //            list.clear();
 //            list.addAll(hs);
 //            // convert list to string
 //            String res = "";
 //            for (IntWritable s : list){
	// 		    res += s.toString() + ",";
	// 		}
	// 		result.set(res);
	// 		context.write(key, result);
	// 	}
	// }

	// public static class auth_coauth_Mapper extends Mapper<Object, Text, IntWritable, IntWritable> {
	// 	private IntWritable author = new IntWritable();
	// 	private IntWritable coauth = new IntWritable();
	// 	protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
	// 		String[] tokens = value.toString().split("\t");
	// 		String[] coauths = tokens[1].split(",");
	// 		for (String a: coauths) {
	// 			for (String c: coauths) {
	// 				int a2 = Integer.parseInt(a);
	// 				int c2 = Integer.parseInt(c);
	// 				if (a2!=c2) {
	// 					author.set(a2); 
	// 					coauth.set(c2);
	// 					context.write(author, coauth); 
	// 				}
					
	// 			}
				
	// 		}
	// 	}
	// }

	// public static class auth_coauth_Reducer extends Reducer <IntWritable, IntWritable, IntWritable, IntWritable> {
	// 	private IntWritable result = new IntWritable();
	// 	ArrayList<IntWritable> list;
	// 	public void reduce(IntWritable key, Iterable <IntWritable> values, Context context) throws IOException, InterruptedException {
	// 		list = new ArrayList<IntWritable>();
	// 		for (IntWritable val: values) {
	// 			list.add(new IntWritable(val.get()));
	// 		}
	// 		// Get unique elements
	// 		HashSet<IntWritable> hs = new HashSet<IntWritable> ();
 //            hs.addAll(list);
 //            list.clear();
 //            list.addAll(hs);
 //            //set result
	// 		result.set(list.size());
	// 		context.write(key, result);
	// 	}
	// }

	public static class topk_Mapper extends Mapper<Object, Text, Text, IntWritable> {
		private IntWritable author = new IntWritable();
		private IntWritable n_coauth = new IntWritable();
		private TreeMap<Integer,Integer> tm = new TreeMap<Integer,Integer>(new Comparator<Integer>(){
                        @Override
                        public int compare(Integer o1, Integer o2) {
                            if(o1>o2){
					            return -1; //descending order
					        } else {
					            return 1;
					        } 
                        }
                    });
		protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			// String[] tokens = value.toString().split("\t");
			// int auth = Integer.parseInt(tokens[0]);
			// int n_auth = Integer.parseInt(tokens[1]);
			// tm.put(n_auth,auth);
			author.set(1);
			context.write(value, author); 	
			}
		// protected void cleanup(Context context) throws IOException, InterruptedException {
		// 	Configuration conf = context.getConfiguration();
		// 	// int k = Integer.parseInt(conf.get("top_k"));
		// 	int k = 5;

		// 	Iterator<Entry<Integer , Integer>> it = tm.entrySet().iterator();
		// 	Entry<Integer , Integer> entry2 = null;
		// 	HashSet<Integer> keys = new HashSet<Integer> ();
		// 	keys.addAll(tm.keySet());

		// 	while(keys.size()>k){
		// 		entry2 = it.next();
		// 		it.remove();
		// 		keys.clear();
		// 		keys.addAll(tm.keySet());
		// 	}
		// 	for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
		// 		author.set(entry.getValue());
		// 		n_coauth.set(entry.getKey());
		// 	    context.write(author, n_coauth); 
		// 	}
		// }	
	}


	public static class topk_Reducer extends Reducer <Text, IntWritable, Text, IntWritable> {
		private IntWritable author = new IntWritable();
		private IntWritable n_coauth = new IntWritable();
		private TreeMap<Integer,Integer> tm = new TreeMap<Integer,Integer>(new Comparator<Integer>(){
                        @Override
                        public int compare(Integer o1, Integer o2) {
                            if(o1>o2){
					            return -1; //descending order
					        } else {
					            return 1;
					        } 
                        }
                    });
		public void reduce(Text key, Iterable <IntWritable> values, Context context) throws IOException, InterruptedException {
			// int auth = Integer.parseInt(key.toString());
			// for (IntWritable val: values) {
			// 	int n_auth = Integer.parseInt(val.toString());
			// 	tm.put(n_auth,auth);
			// }
			for (IntWritable val: values) {
				author.set(1);
				context.write(key, val); 
			}
		}
		// protected void cleanup(Context context) throws IOException, InterruptedException {
		// 	Configuration conf = context.getConfiguration();
		// 	// int k = Integer.parseInt(conf.get("top_k"));
		// 	int k = 5;

		// 	Iterator<Entry<Integer , Integer>> it = tm.entrySet().iterator();
		// 	Entry<Integer , Integer> entry2 = null;
		// 	HashSet<Integer> keys = new HashSet<Integer> ();
		// 	keys.addAll(tm.keySet());

		// 	while(keys.size()>k){
		// 		entry2 = it.next();
		// 		it.remove();
		// 		keys.clear();
		// 		keys.addAll(tm.keySet());
		// 	}
		// 	for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
		// 		author.set(entry.getValue());
		// 		n_coauth.set(entry.getKey());
		// 	    context.write(author, n_coauth); 
		// 	}
		// }	
	}

	

	public static void main(String[] args) throws Exception {
		// Path TEMP_PATH = new Path("/user/fzapfack/temp1");
		// Configuration conf = new Configuration();
		// Job job = Job.getInstance(conf, "job1");
		// job.setJarByClass(Q1.class);
		// job.setMapperClass(doc_auth_Mapper.class);
		// // job.setCombinerClass(doc_auths_Reducer.class);
		// job.setReducerClass(doc_auths_Reducer.class);
		// job.setMapOutputKeyClass(IntWritable.class);
		// job.setMapOutputValueClass(IntWritable.class);
		// job.setOutputKeyClass(IntWritable.class);
		// job.setOutputValueClass(Text.class);
		// FileInputFormat.addInputPath(job, new Path(args[0]));
		// // job.setOutputFormatClass(SequenceFileOutputFormat.class);
		// FileOutputFormat.setOutputPath(job, TEMP_PATH);
		// job.waitForCompletion(true);
		// // Chain using others jobs
		// Configuration conf2 = new Configuration();
		// Job job2 = Job.getInstance(conf2, "job2");
		// job2.setJarByClass(Q1.class);
		// job2.setMapperClass(auth_coauth_Mapper.class);
		// job2.setReducerClass(auth_coauth_Reducer.class);
		// job2.setMapOutputKeyClass(IntWritable.class);
		// job2.setMapOutputValueClass(IntWritable.class);
		// job2.setOutputKeyClass(IntWritable.class);
		// job2.setOutputValueClass(IntWritable.class);
		// // job2.setInputFormatClass(SequenceFileInputFormat.class);
		// FileInputFormat.addInputPath(job2, TEMP_PATH);
		Path TEMP_PATH2 = new Path("/user/fzapfack/temp2");
		// FileOutputFormat.setOutputPath(job2, TEMP_PATH2);
		// job2.waitForCompletion(true);



		// Top-k job
		
		Configuration conf3 = new Configuration();
		Job job3 = Job.getInstance(conf3, "job3");
		// conf3.set("top_k", args[2]);
		conf3.set("mapred.reduce.tasks", "1");
		job3.setJarByClass(Q1.class);
		job3.setMapperClass(topk_Mapper.class);
		job3.setReducerClass(topk_Reducer.class);
		job3.setMapOutputKeyClass(IntWritable.class);
		job3.setMapOutputValueClass(IntWritable.class);
		job3.setOutputKeyClass(IntWritable.class);
		job3.setOutputValueClass(IntWritable.class);
		// job2.setInputFormatClass(SequenceFileInputFormat.class);
		FileInputFormat.addInputPath(job3, TEMP_PATH2);
		FileOutputFormat.setOutputPath(job3, new Path(args[1]));
		// TEMP_PATH.getFileSystem(conf).delete(TEMP_PATH);
		// TEMP_PATH2.getFileSystem(conf).delete(TEMP_PATH2);

		System.exit(job3.waitForCompletion(true) ? 0 : 1);
	}
}