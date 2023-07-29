#!/bin/bash


read_config() {
	variable_name=$1
	grep "$variable_name" config.ini | cut -d'=' -f2 | tr -d '[:space:]'
}

dt=$(read_config dt)
termi_t=$(read_config terminate_time)
cur_cnt=$(read_config loop_index)
cur_t=$(echo "scale=2; ${cur_cnt}*${dt}" |bc -l )
termi_t=$(echo "scale=2; ${termi_t}-${dt}" |bc -l )
while [ $(echo "$cur_t < $termi_t" | bc -l) ]
do
	cur_t=$(echo "scale=2; ${cur_cnt}*${dt}" |bc -l )
	cur_pid=$!
	find_pid=`ps -e | grep "$cur_pid" | grep 'python qite_run_time_script.py'`
	if [ "$find_pid" == "" ];then
		cur_data_num=$(read_config data_index)
		save_data_num=$(($cur_data_num+1))
		nohup python qite_run_time_script.py &> ../log/data_${save_data_num}_log.txt&
		cur_pid=$!
		sleep 100
	fi
done
