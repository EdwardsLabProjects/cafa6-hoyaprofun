#!.venv/bin/python

import sys
import steps

print("\n[0/6] Load configuration and initial files...")

CONFIG = steps.configuration(sys.argv[1:])

result_file = CONFIG["RESULT"]
model_pred_file = CONFIG["MODEL_RESULT"]

if CONFIG['MERGE_WITH_GOA']:
    result_files = [ model_pred_file, result_file ]
else:
    results_files = [ result_file ]

steps.write_submission_plot(CONFIG,*result_files,outfile="confidence.png")

train_truth = steps.load_train_terms_ground_truth(CONFIG)
steps.write_precall_plot(CONFIG,train_truth,*result_files,outfile="train_pr.png")

ground_truth = steps.load_ground_truth(CONFIG)
steps.write_precall_plot(CONFIG,ground_truth,*result_files,outfile="ground_pr.png")
steps.write_precall_plot(CONFIG,ground_truth,*result_files,
                         ignore=train_truth,outfile="ground_no_train_pr.png")



