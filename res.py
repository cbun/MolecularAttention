import numpy as np
import matplotlib.pyplot as plt
from regression_enrichment_surface import regression_enrichment_surface as rds
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
import argparse
from scipy import stats

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input file')
    parser.add_argument('output', type=str, help='output dir')
    parser.add_argument('name', type=str, help='target name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    out = np.load(args.input)
    values, preds = out[0], out[1]
    quant = np.quantile(values, 0.90)
    values_bin = [0 if val < quant else 1 for val in values]
    preds_bin = [0 if val < quant else 1 for val in preds]
    print("pearson", stats.pearsonr(values, preds))
    print("r2", r2_score(values, preds))
    print("mae", mean_absolute_error(values, preds))
    print("precision, recall" ,precision_score(values_bin, preds_bin), recall_score(values_bin, preds_bin))
    fpr, tpr, thresholds = roc_curve(values_bin, preds_bin)
    print("auc", auc(fpr, tpr))
    rds_model = rds.RegressionEnrichmentSurface(percent_min=-4)
    print(values.shape, preds.shape)
    rds_model.compute(values.flatten(), preds.flatten(), samples=50)
    rds_model.plot(save_file= args.output +"/res-"+ args.name + ".png",
                   title='Regression Enrichment Surface ' + args.name + ' image model')
