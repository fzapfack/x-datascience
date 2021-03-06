
import scala.io.Source

object QuickSort {
	def quick(xs: Array[Int]): Array[Int] = {
		if (xs.length <= 1) xs
		else {
		val pivot = xs(xs.length / 2)
		Array.concat(quick(xs filter (_ < pivot)),
		xs filter (_ == pivot), quick(xs filter (_ > pivot)))
		}
	}

	def main(args: Array[String]) {
		val xs = Array.fill(100){scala.util.Random.nextInt(99)}
		println(quick(xs))
	}    
}
