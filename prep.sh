
	grep -o 'name [^;]*' meters.txt | cut -d' ' -f2 |awk '{print "\"" $1 "\":[\"measured_voltage_1\",\"measured_voltage_2\",\"measured_voltage_N\"],"}'
