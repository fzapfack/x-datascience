// import java.util.TreeMap;
// import java.util.Map;
// import java.util.Comparator;
import java.util.Map.Entry;
import java.util.*;

public class test {

	
	// public static TreeMap<Integer, Integer> ToRecordMap = new TreeMap<Integer , Integer>(new comp());
    public static void main(String[] args) {
        // Prints "Hello, World" to the terminal window.
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
        tm.put(5,6);
        tm.put(5,3);
        tm.put(1,2);
        tm.put(3,4);
        tm.put(3,5);
        for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
		     System.out.println("Key: " + entry.getKey() + ". Value: " + entry.getValue());
		}
    	// Comp a = new Comp();
    	// TreeMap 
  //       String res="";
		// int[] a = {5,4,3};
		// for(int i : a) {
		// 	res +="#";
		// 	res += Integer.toString(i);
		// }
		Iterator<Entry<Integer , Integer>> it = tm.entrySet().iterator();
		Entry<Integer , Integer> entry2 = null;
		// int counter = 0;
		HashSet<Integer> keys = new HashSet<Integer> ();
		keys.addAll(tm.keySet());
		// Set<Integer> keys = tm.keySet();
		// for (int k:keys){
		// 	System.out.println(k);
		// }
		while(keys.size()>2){
			entry2 = it.next();
			it.remove();
			keys.clear();
			keys.addAll(tm.keySet());
		}
		System.out.println("sdsd");
		String s = "";
		for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
		     // System.out.println("Key: " + entry.getKey() + ". Value: " + entry.getValue());
			s += "<"+ entry.getKey() +", "+ entry.getValue()+">";
			s += "\t"; 
		}
		System.out.println(s);
		// TreeMap tm2 = new TreeMap<Integer,Integer>(tm);
		// System.out.println("Lowest key Stored in Java TreeMap is : "
  //                                                + tm2.lastKey());
		
    }

}