import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from classifiers.feature_extraction import Features


datasetB_nn = pd.read_csv("data_set/raw/setB_raw.csv")
datasetA_nn = pd.read_csv("data_set/raw/setA_raw.csv")

extractor_tmp = Features()
header_nn = extractor_tmp.nn_input_feature + ["Label"]
header_rule = extractor_tmp.rule_input_feature + ["Label"]
extractor_tmp.close()

for col in header_nn:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=datasetB_nn, x=col, hue="Label", fill=True, 
                common_norm=False, alpha=0.4, warn_singular=False)
    plt.title(f"Distribution of {col} by Label")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( f"data_set/diagnostic/raw_plots_nn/{col}_by_label.png")
    plt.close()

for col in header_nn:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=datasetA_nn, x=col, hue="Label", fill=True, 
                common_norm=False, alpha=0.4, warn_singular=False)
    plt.title(f"Distribution of {col} by Label")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( f"data_set/diagnostic/raw_plots_nn_A/{col}_by_label.png")
    plt.close()

datasetB_rul = pd.read_csv("data_set/raw/subdataB_raw.csv")
datasetA_rul = pd.read_csv("data_set/raw/subdataA_raw.csv")

for col in header_rule:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=datasetB_rul, x=col, hue="Label", fill=True, 
                common_norm=False, alpha=0.4, warn_singular=False)
    plt.title(f"Distribution of {col} by Label")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( f"data_set/diagnostic/raw_plots_rul/{col}_by_label.png")
    plt.close()

for col in header_rule:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=datasetA_rul, x=col, hue="Label", fill=True, 
                common_norm=False, alpha=0.4, warn_singular=False)
    plt.title(f"Distribution of {col} by Label")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig( f"data_set/diagnostic/raw_plots_rul_A/{col}_by_label.png")
    plt.close()







scale_needed = ["yaw", "pitch", "roll", "face_ratio", "body_yaw", "avg_body_movement"]
scaler = MinMaxScaler()
datasetB_nn_scaled = datasetB_nn.copy()
datasetB_nn_scaled[scale_needed] = scaler.fit_transform(datasetB_nn_scaled[scale_needed])
joblib.dump(scaler, "data_set/minmax_scaler.pkl")
datasetB_nn_scaled.to_csv("data_set/setB_scaled.csv", index=False)

datasetA_nn_scaled = datasetA_nn.copy()
datasetA_nn_scaled[scale_needed] = scaler.transform(datasetA_nn_scaled[scale_needed])
datasetA_nn_scaled.to_csv("data_set/setA_scaled.csv", index=False)





# from sklearn.feature_selection import mutual_info_classif
# X_nn = datasetB_nn_scaled[header_nn[:-1]]
# y_nn = datasetB_nn_scaled["Label"]
# mi_scores = mutual_info_classif(X_nn, y_nn, discrete_features='auto')
# mi_series = pd.Series(mi_scores, index=header_nn[:-1])
# mi_series = mi_series.sort_values(ascending=False)
# print(mi_series)


# corr_matrix = datasetB_nn_scaled[header_nn[:-1]].corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
# plt.title("Correlation Matrix (Neural Features)")
# plt.show()