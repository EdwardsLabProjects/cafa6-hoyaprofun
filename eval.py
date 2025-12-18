#!.venv/bin/python

import sys
import steps
from util import getfile

CONFIG = steps.configuration(sys.argv[1:])

result_file = getfile(CONFIG["RESULT"])
model_pred_file = getfile(CONFIG["MODEL_RESULT"])

steps.run_cafa6_eval(CONFIG,model_pred_file,result_file)
steps.cafa6_plots(CONFIG)























