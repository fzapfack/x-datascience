/* LineCount.scala */

object LineCount {
  def main(args: Array[String]) {
    val inputFile = "leonardo.txt" // Should be some file on your system
    val src = scala.io.Source.fromFile(inputFile)
    val counter = src.getLines().map(line => 1).sum
       
    println("Number of lines in file: "+counter)
  }
}
