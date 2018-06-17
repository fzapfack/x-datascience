for folder in $(ls -d */); 
do 
	echo "Compiling project: "${folder%%/};
        (cd ./$folder; sbt compile)
        echo "**********************************" 
done
