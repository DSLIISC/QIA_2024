#!/bin/bash
result_location="Results"
SQIA_codefile="GS_Paper-All_Experiments.py"
OQIA_codefile="QAOA_Paper-All_Experiments.py"

pid_list1=(5 6 7 72)
pid_list2=(5 6 7 71)

for i in "${pid_list1[@]}"; do
	echo Executing python $SQIA_codefile -pid $i -bp $result_location --no-noisy 
	python $SQIA_codefile -pid $i -bp $result_location --no-noisy
done

for i in "${pid_list2[@]}"; do
	echo Executing python $OQIA_codefile -pid $i -bp $result_location --no-noisy --inc
	python $OQIA_codefile -pid $i -bp $result_location --no-noisy --inc
done
