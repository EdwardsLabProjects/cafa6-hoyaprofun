#!.venv/bin/python

import sys
import steps

CONFIG = steps.configuration(sys.argv[1:])

result_file = CONFIG["RESULT"]
model_pred_file = CONFIG["MODEL_RESULT"]

steps.run_cafa6_eval(CONFIG,model_pred_file,result_file)























