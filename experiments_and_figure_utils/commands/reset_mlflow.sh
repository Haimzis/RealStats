kill $(ps aux | grep mlflow | awk '{print $2}')
mlflow ui