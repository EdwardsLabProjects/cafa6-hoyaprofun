#!.venv/bin/python

import sys
import steps
from util import getfile

CONFIG = steps.configuration(sys.argv[1:])

result_file = getfile(CONFIG["RESULT"],"Result file")
model_pred_file = getfile(CONFIG["MODEL_RESULT"],"Model result file")

steps.run_cafa6_eval(CONFIG,model_pred_file,result_file)
steps.cafa6_plots(CONFIG)























