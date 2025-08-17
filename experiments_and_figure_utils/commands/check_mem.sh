ps -u haimzis -o rss | awk '{sum+=$1} END {print sum/1024/1024 " GB"}'
ls pkls/ | wc -l
