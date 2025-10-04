ps aux | grep executor |  awk '{print $2}' | xargs -I {} kill {}
ps aux | grep test_processing.py |  awk '{print $2}' | xargs -I {} kill {}
ps aux | grep rigid |  awk '{print $2}' | xargs -I {} kill {}
ps aux | grep wavelets_detector/bin/python |  awk '{print $2}' | xargs -I {} kill {}