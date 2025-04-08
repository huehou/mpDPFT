#!/bin/bash
myVariable=`ps -C mpDPFT -o pid=`
#echo $myVariable
cd /proc/
tail --lines=10000 -f $myVariable/fd/1
#ps -e S #to view sleeping processes