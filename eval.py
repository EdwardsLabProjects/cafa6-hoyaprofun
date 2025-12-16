#!.venv/bin/python

import sys
import steps

print("\n[0/6] Load configuration and initial files...")

CONFIG = steps.configuration(sys.argv[1:])

result_file = CONFIG["RESULT"]
model_pred_file = CONFIG["MODEL_RESULT"]

steps.cafa_eval(CONFIG,model_pred_file)
steps.cafa_eval(CONFIG,result_file)
























