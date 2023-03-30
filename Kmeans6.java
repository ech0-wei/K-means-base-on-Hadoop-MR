import java.awt.geom.Point2D;
import java.io.IOException;
import java.util.*;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
public class Kmeans6 {

    public static class PointsMapper extends Mapper<LongWritable, Text, IntWritable, Text> {

        // centroids : Linked-list/arraylike
        LinkedList<Point2D> centers = new LinkedList<Point2D>();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {

            super.setup(context);
            Configuration conf = context.getConfiguration();

            // retrive file path
            Path centroids = new Path(conf.get("centroid.path"));

            // create a filesystem object
            FileSystem fs = FileSystem.get(conf);

            // create a file reader
            SequenceFile.Reader reader = new SequenceFile.Reader(fs, centroids, conf);

            // read centroids from the file and store them in a centroids variable
            IntWritable key = new IntWritable();
            Text value = new Text();
            while (reader.next(key, value)) {

                String temp = value.toString();
                temp = temp.replace("Point2D.Double[", "");
                temp = temp.replace("]", "");
                //Store the x-axis y-axis
                String[] tempXY = temp.split(",");
                // Convert to Point2D
                double X = Double.parseDouble(tempXY[0]);
                double Y = Double.parseDouble(tempXY[1]);
                // Prepare centers
                centers.add(new Point2D.Double(X, Y));
            }
            reader.close();

        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            // input -> key: charater offset, value -> a point (in Text)
            // Read the point from value
            String temp = value.toString();
            String[] tempPoint = temp.split(",");
            // create x,y and point
            double X = Double.parseDouble(tempPoint[0]);
            double Y = Double.parseDouble(tempPoint[1]);
            Point2D point = new Point2D.Double(X, Y);
            // write logic to assign a point to a centroid
            // Set initial distance
            double minDist = 9999;
            int centerID = 0;
            // For each center, and assign the groupID(AKA centerID) for point
            for (Point2D center : centers) {
                if (center.distance(point) < minDist) {
                    minDist = center.distance(point);
                    centerID = centers.indexOf(center);
                }
            }
            // emit key (centroid id/centroid) and value (point)
            IntWritable outK = new IntWritable(centerID);
            Text outV = value;
            context.write(outK, outV);
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {

        }

    }

    public static class PointsReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        LinkedList<Point2D> newCenters = new LinkedList<Point2D>();

        public static enum Counter {
            CONVERGED
        }
        // new_centroids (variable to store the new centroids

        @Override
        public void setup(Context context) {

        }

        @Override
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // Input: key -> centroid id/centroid , value -> list of points
            ArrayList<Point2D> pointList = new ArrayList<Point2D>();
            double newX = 0;
            double newY = 0;
            for (Text value : values) {
                // Read the point from value
                String temp = value.toString();
                String[] tempPoint = temp.split(",");
                // create x,y and point
                double X = Double.parseDouble(tempPoint[0]);
                double Y = Double.parseDouble(tempPoint[1]);
                Point2D point = new Point2D.Double(X, Y);
                pointList.add(point);
                // Prepare for new center calculate
                newX = newX + X;
                newY = newY + Y;
            }

            // calculate the new centroid
            double centerX = newX / pointList.size();
            double centerY = newY / pointList.size();
            Point2D center = new Point2D.Double(centerX, centerY);
            // new_centroids.add() (store updated cetroid in a variable)
            newCenters.add(center);
            // Get the id of center
            int centerID = newCenters.indexOf(center);
            // Get the x-axis,y-axis
            String centerXY = center.toString();
            centerXY = centerXY.replace("Point2D.Double", "");

            context.write(new IntWritable(centerID), new Text(centerXY));
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            // BufferedWriter
            Configuration conf = context.getConfiguration();
            Path centroids = new Path(conf.get("centroid.path"));
            FileSystem fs = FileSystem.get(conf);
            SequenceFile.Writer newCenterwriter = SequenceFile.createWriter(fs, conf, centroids, IntWritable.class, Text.class);
            // delete the old centroids
            // The append method will overwrite the file
            // write the new centroids
            for (Point2D centroid : newCenters) {
                IntWritable centroidID = new IntWritable(newCenters.indexOf(centroid) + 1);
                Text centroidText = new Text(centroid.toString());
                newCenterwriter.append(centroidID, centroidText);

            }
            newCenterwriter.close();
        }

    }

    public static void main(String[] args) throws Exception {

        Configuration conf = new Configuration();

        Path center_path = new Path("centroid/cen.seq");
        conf.set("centroid.path", center_path.toString());

        FileSystem fs = FileSystem.get(conf);

        if (fs.exists(center_path)) {
            fs.delete(center_path, true);
        }

        final SequenceFile.Writer centerWriter = SequenceFile.createWriter(fs, conf, center_path,
                IntWritable.class,Text.class);

        centerWriter.append(new IntWritable(1),new Text("50.197031637442876,32.94048164287042"));
        centerWriter.append(new IntWritable(2),new Text("43.407412339767056,6.541037020010927"));
        centerWriter.append(new IntWritable(3),new Text("1.7885358732482017,19.666057053079573"));
        centerWriter.append(new IntWritable(4),new Text("32.6358540480337,4.03843047564191"));
        centerWriter.append(new IntWritable(5),new Text("11.959317048618196,18.52941355338217"));
        centerWriter.append(new IntWritable(6),new Text("9.293805975483108,8.886169685374657"));

        centerWriter.close();
        int itr = 0;
        while (itr < 20) {
            // config
            // job
            Job job = new Job(conf, "kmean_iteration_" + (itr + 1));

            // set the job parameters
            job.setJarByClass(Kmeans6.class);

            job.setMapperClass(PointsMapper.class);
            job.setReducerClass(PointsReducer.class);

            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(Text.class);

            if (fs.exists(new Path(args[1]))) {
                fs.delete(new Path(args[1]), true);
            }

            FileInputFormat.setInputPaths(job, new Path(args[0]));
            FileOutputFormat.setOutputPath(job, new Path(args[1]));

            job.waitForCompletion(true);

            // set the job parameters


            // Print result
            SequenceFile.Reader reader = new SequenceFile.Reader(fs, center_path, conf);

            System.out.println("\n" + "Iteration number: " + (itr + 1));


            // print the centroids (final result)
            IntWritable key = new IntWritable();
            Text value = new Text();
            while (reader.next(key, value)) {

                String centroid = value.toString();
                String ID = key.toString();
                centroid = centroid.replace("Point2D.Double","");
                System.out.println("\n"+ID+","+centroid+"\n");
            }

            reader.close();
            // itr ++
            itr++;
        }

        // read the centroid file from hdfs and print the centroids (final result
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, center_path, conf);

        System.out.println("\nFinal Result:");

        // print the centroids (final result)
        IntWritable key = new IntWritable();
        Text value = new Text();
        while (reader.next(key, value)) {

            String centroid = value.toString();
            String ID = key.toString();
            centroid = centroid.replace("Point2D.Double","");
            System.out.println("\n"+ID+","+centroid+"\n");
        }
        reader.close();
    }
}
