import sys, os
import json
import logging
import run_expts_nonBERT as run_expt
logger = logging.getLogger('run_expt')
formatter = logging.Formatter('%(asctime)s|%(process)d|%(levelname)s|%(funcName)s|%(message)s')

if len(sys.argv) <= 1:
   print("No arguments were given")
else:
   json_file_path = sys.argv[1]
   print(f"Arg recd: {json_file_path}")

   # with open(json_file_path) as f:
   #    logfile = json.loads(f.read())['logfile']
   # file_handler = logging.FileHandler(logfile)
   # file_handler.setLevel(logging.INFO)
   # file_handler.setFormatter(formatter)
   # logger.addHandler(file_handler)

   run_expt.run_expt_nomp(json_file_path)