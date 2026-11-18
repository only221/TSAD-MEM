import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.mitric.vus_metrics import metricor
from utils.mitric.affiliation.generics import convert_vector_to_events
from utils.mitric.affiliation.metrics import pr_from_events

class Metrics:
    
    def __init__(self, a=1, probability=True, bias="flat"):
        self.a = a
        self.probability = probability
        self.bias = bias
    
    def adjust_predicts(self, actual, predicted):

        predicted = predicted.copy()
        anomaly_state = False
        for i in range(len(actual)):
            if actual[i] == 1 and predicted[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, -1, -1):
                    if actual[j] == 0:
                        break
                    else:
                        if predicted[j] == 0:
                            predicted[j] = 1
                for j in range(i, len(actual)):
                    if actual[j] == 0:
                        break
                    else:
                        if predicted[j] == 0:
                            predicted[j] = 1
            elif actual[i] == 0:
                anomaly_state = False
            if anomaly_state:
                predicted[i] = 1
        return predicted


    def evaluate_all_metrics_label(self, y_true, y_pred_labels):
        results = {}
        metricor_grader = metricor()
        

        results['Accuracy'] = accuracy_score(y_true, y_pred_labels)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred_labels, zero_division=0
        )
        results['Precision'] = precision[1] if len(precision) > 1 else 0
        results['Recall'] = recall[1] if len(recall) > 1 else 0
        results['F1_Score'] = f1[1] if len(f1) > 1 else 0

        adjusted_preds = self.adjust_predicts(y_true, y_pred_labels)
        results['Adjust_Accuracy'] = accuracy_score(y_true, adjusted_preds)
        adj_precision, adj_recall, adj_f1, _ = precision_recall_fscore_support(
            y_true, adjusted_preds, zero_division=0
        )
        results['Adjust_Precision'] = adj_precision[1] if len(adj_precision) > 1 else 0
        results['Adjust_Recall'] = adj_recall[1] if len(adj_recall) > 1 else 0
        results['Adjust_F1_Score'] = adj_f1[1] if len(adj_f1) > 1 else 0
        

        try:
            (
            AUC_ROC, Precision, Recall, F, Rrecall, ExistenceReward, 
            OverlapReward, Rprecision, RF, Precision_at_k
            ) = metricor_grader.metric_new(y_true, y_pred_labels, plot_ROC=False)
        
            results['Range_Recall'] = Rrecall
            results['Range_Precision'] = Rprecision  
            results['Range_F1'] = RF
        except Exception as e:
            print(f"Warning: metricor calculation failed: {e}")
            results['Range_Recall'] = 0.0
            results['Range_Precision'] = 0.0
            results['Range_F1'] = 0.0
        

        try:
            results['Precision_at_k'] = Precision_at_k
        except:

            k = int(np.sum(y_true))
            if k > 0:

                results['Precision_at_k'] = np.sum(y_true * y_pred_labels) / k
            else:
                results['Precision_at_k'] = 0.0
        

        try:
            events_pred = convert_vector_to_events(y_pred_labels)
            events_label = convert_vector_to_events(y_true)
            Trange = (0, len(y_pred_labels))
            
            result = pr_from_events(events_pred, events_label, Trange)
            P = result['precision']
            R = result['recall']
            F = 2 * P * R / (P + R + 1e-8)
            
            results['Affiliation_Precision'] = P
            results['Affiliation_Recall'] = R
            results['Affiliation_F1'] = F
        except Exception as e:
            print(f"Warning: affiliation calculation failed: {e}")
            results['Affiliation_Precision'] = 0.0
            results['Affiliation_Recall'] = 0.0
            results['Affiliation_F1'] = 0.0
        
        return results