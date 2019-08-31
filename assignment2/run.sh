#1/bin/sh
if [ "$1" = "1" ]
then 
		python naive1.py $2 $3 $4
		#python sum.py $2 $3 $4
elif [ "$1" = "2" ]
then 
		#python svm.py $2 $3 $4 $5
		python svm1.py $2 $3 $4 $5
fi				
		
