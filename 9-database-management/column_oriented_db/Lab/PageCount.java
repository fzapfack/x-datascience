import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
public class PageCount {
public static class ProjectionMapper extends Mapper<Object, Text, IntWritable, IntWritable> {
private IntWritable venue = new IntWritable();
private IntWritable pages = new IntWritable();protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
// Our file contains space-limited values: paper_id, paper_name, venue_id, pages, url
// We project out: venue_id, pages
String[] tokens = value.toString().split("\t");
if (tokens.length == 5 && tokens[2].matches("^-?\\d+$")) {
venue.set(Integer.parseInt(tokens[2])); // venue_id
if (tokens[3].contains("-")) {
String[] range = tokens[3].split("-");
if (range.length == 2 && range[0].trim().matches("^-?\\d+$") && range[1].trim().matches("^-?\\d+$")) {
int length = Integer.parseInt(range[1].trim()) - Integer.parseInt(range[0].trim());
pages.set(length);
context.write(venue, pages);
}
}
}
}
}
public static class IntSumReducer extends Reducer <IntWritable, IntWritable, IntWritable, IntWritable> {
private IntWritable result = new IntWritable();
public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
throws IOException, InterruptedException {
int sum = 0;
for (IntWritable val: values) {
sum += val.get();
}
result.set(sum);
context.write(key, result);
}
}
public static void main(String[] args) throws Exception {
Configuration conf = new Configuration();
Job job = Job.getInstance(conf, "page count");
job.setJarByClass(PageCount.class);
job.setMapperClass(ProjectionMapper.class);
job.setCombinerClass(IntSumReducer.class);
job.setReducerClass(IntSumReducer.class);
job.setOutputKeyClass(IntWritable.class);
job.setOutputValueClass(IntWritable.class);
FileInputFormat.addInputPath(job, new Path(args[0]));
FileOutputFormat.setOutputPath(job, new Path(args[1]));
System.exit(job.waitForCompletion(true) ? 0 : 1);
}
}