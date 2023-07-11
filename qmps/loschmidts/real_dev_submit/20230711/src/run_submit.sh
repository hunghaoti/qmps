#!/bin/bash

device_name=`grep '^device_name:' para_set.txt | sed 's/^.*://'`
data_path=`grep '^data_path:' para_set.txt | sed 's/^.*://'`
dt=`grep '^dt:' para_set.txt | sed 's/^.*://'`
termi_t=`grep '^termi_t:' para_set.txt | sed 's/^.*://'`
cur_cnt=`cat cur_stage/cnt.txt`

function re_gen_code(){
	cur_data_num=`cat cur_stage/cur_data.txt`
	load_data_num=$cur_data_num
	save_data_num=$(($cur_data_num+1))
	mkdir -p $data_path/data$save_data_num/params
	cur_cnt=`cat cur_stage/cnt.txt`
	is_init=$(echo "$cur_cnt == 0" | bc -l)
	if [ $is_init -eq 1 ]
	then
		init_flag1=""
		init_flag2="#"
	else 
		init_flag1="#"
		init_flag2=""
	fi
	
	cat temp_qite_run_time_script.py | sed 's+%device_name+'"$device_name"'+g' > qite_run_time_script.py
	cat temp_qite_qiskit.py \
		| sed 's+%data_path+'"$data_path"'+g' \
		| sed 's+%cnt+'"$cur_cnt"'+g' \
		| sed 's+%init_flag1+'"$init_flag1"'+g' \
		| sed 's+%init_flag2+'"$init_flag2"'+g' \
		| sed 's+%data_load_num+'"$load_data_num"'+g' \
		| sed 's+%data_save_num+'"$save_data_num"'+g' \
		| sed 's+%dt+'"$dt"'+g' \
		| sed 's+%termi_t+'"$termi_t"'+g' \
		> qite_qiskit.py
	cur_data_num=$(($cur_data_num+1))
	echo $cur_data_num > cur_stage/cur_data.txt
}


cur_t=$(echo "scale=2; ${cur_cnt}*${dt}" |bc -l )
termi_t=$(echo "scale=2; ${termi_t}-${dt}" |bc -l )
all_done=$(echo "$cur_t >= $termi_t" |bc -l)
still_run=0
while [ $all_done == 0 ]
do
	cur_t=$(echo "scale=2; ${cur_cnt}*${dt}" |bc -l )
	all_done=$(echo "$cur_t >= $termi_t" |bc -l)
	cur_pid=$!
	find_pid=`ps -e | grep $cur_pid | grep ipython`
	if [ "$find_pid" == "" ];then
		re_gen_code
		nohup ipython qite_run_time_script.py &> ../log/data_${save_data_num}_log.txt&
		cur_pid=$!
		sleep 100
	fi
	
done
