#!.venv/bin/python

import sys
import steps

print("\n[0/6] Load configuration and initial files...")

CONFIG = steps.configuration(sys.argv[1:])

result_file = CONFIG["RESULT"]
model_pred_file = CONFIG["MODEL_RESULT"]

if CONFIG['MERGE_WITH_GOA']:
    steps.write_submission_plot(CONFIG,model_pred_file,result_file,outfile="confidence.png")

    train_truth = steps.load_train_terms_ground_truth(CONFIG)
    steps.write_precall_plot(CONFIG,train_truth,model_pred_file,result_file,outfile="train_pr.png")

else:
    steps.write_submission_plot(CONFIG,result_file,outfile="confidence.png")

    train_truth = steps.load_train_terms_ground_truth(CONFIG)
    steps.write_precall_plots(CONFIG,train_truth,result_file,outfile="train_pr.png")






