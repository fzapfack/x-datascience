for folder in $(ls -d */); 
do 
	echo "Cleaning project: "${folder%%/};
        (cd ./$folder; sbt clean)
        echo "**********************************" 
done
