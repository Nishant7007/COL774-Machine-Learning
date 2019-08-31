
#!/bin/sh
if [ "$1" = "1" ]
then
        python question1.py $2 $3 $4 $5
elif [ "$1" = "2" ]
then
        python question2.py $2 $3 $4
elif [ "$1" = "3" ]
then
        python question_3.py $2 $3
elif [ "$1" = "4" ]
then
        python question4.py $2 $3 $4
fi