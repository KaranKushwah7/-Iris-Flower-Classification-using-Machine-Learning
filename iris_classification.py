# ============================================================
#  🌸 IRIS FLOWER CLASSIFICATION — ML INTERNSHIP PROJECT
#  Author  : Machine Learning Internship
#  Dataset : Iris Dataset (150 samples, 3 species)
#  Models  : Logistic Regression, KNN, Decision Tree,
#             Random Forest, SVM, Naive Bayes
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier, plot_tree
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.naive_bayes     import GaussianNB

import os, sys

# ─────────────────────────────────────────────
# 0.  OUTPUT FOLDER
# ─────────────────────────────────────────────
OUT = "."
os.makedirs(OUT, exist_ok=True)

COLORS  = ["#4CAF50", "#2196F3", "#FF5722"]
SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  🌸  IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 60)

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"])
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
target   = "Species"

print(f"\n📊 Dataset Shape  : {df.shape}")
print(f"📋 Features       : {features}")
print(f"🌺 Target Classes : {df[target].unique().tolist()}")
print(f"\n{df.head(5).to_string(index=False)}")

# ─────────────────────────────────────────────
# 2.  EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  📈  EXPLORATORY DATA ANALYSIS")
print("─" * 60)

# 2-a  Basic stats
print(f"\n🔍 Missing values  : {df.isnull().sum().sum()}")
print(f"📊 Class balance  :")
print(df[target].value_counts().to_string())
print(f"\n📐 Descriptive Statistics:")
print(df[features].describe().round(2).to_string())

# 2-b  EDA Figure — 4 sub-plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Iris Flower — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=0.98)

#  i. Pairplot-style scatter: petal length vs petal width
ax = axes[0, 0]
for sp, col in zip(SPECIES, COLORS):
    sub = df[df[target] == sp]
    ax.scatter(sub["PetalLengthCm"], sub["PetalWidthCm"],
               c=col, label=sp.replace("Iris-", ""), alpha=0.8, s=60, edgecolors="white")
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width  (cm)")
ax.set_title("Petal Length vs Petal Width")
ax.legend()
ax.grid(alpha=0.3)

#  ii. Sepal scatter
ax = axes[0, 1]
for sp, col in zip(SPECIES, COLORS):
    sub = df[df[target] == sp]
    ax.scatter(sub["SepalLengthCm"], sub["SepalWidthCm"],
               c=col, label=sp.replace("Iris-", ""), alpha=0.8, s=60, edgecolors="white")
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Sepal Width  (cm)")
ax.set_title("Sepal Length vs Sepal Width")
ax.legend()
ax.grid(alpha=0.3)

#  iii. Box plot — all features
ax = axes[1, 0]
df_melt = df.melt(id_vars=target, value_vars=features,
                  var_name="Feature", value_name="Value")
palette = {sp: col for sp, col in zip(SPECIES, COLORS)}
sns.boxplot(data=df_melt, x="Feature", y="Value", hue=target,
            palette=palette, ax=ax, linewidth=1.0)
ax.set_title("Feature Distribution by Species")
ax.set_xlabel("")
ax.set_xticklabels([f.replace("Cm", "") for f in features], rotation=15)
ax.legend(title="", fontsize=8)
ax.grid(axis="y", alpha=0.3)

#  iv. Correlation heatmap
ax = axes[1, 1]
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Heatmap")
ax.set_xticklabels([f.replace("Cm", "") for f in features], rotation=20)
ax.set_yticklabels([f.replace("Cm", "") for f in features], rotation=0)

plt.tight_layout()
fig.savefig(f"{OUT}/1_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅  EDA plot saved.")

# ─────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  ⚙️   PREPROCESSING")
print("─" * 60)

X = df[features].values
le = LabelEncoder()
y = le.fit_transform(df[target])            # 0=setosa, 1=versicolor, 2=virginica

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"  Train size : {len(X_train)} samples")
print(f"  Test  size : {len(X_test)}  samples")
print(f"  Scaling    : StandardScaler applied")

# ─────────────────────────────────────────────
# 4.  TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🤖  TRAINING 6 CLASSIFIERS")
print("─" * 60)

models = {
    "Logistic Regression" : LogisticRegression(max_iter=200, random_state=42),
    "K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=5),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    "Naive Bayes"         : GaussianNB(),
}

results = {}
print(f"\n  {'Model':<28} {'Train Acc':>10} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>8}")
print("  " + "-" * 72)

for name, model in models.items():
    # fit on scaled data
    model.fit(X_train_s, y_train)
    tr_acc = accuracy_score(y_train, model.predict(X_train_s))
    te_acc = accuracy_score(y_test,  model.predict(X_test_s))
    cv     = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")
    results[name] = {
        "model"    : model,
        "train_acc": tr_acc,
        "test_acc" : te_acc,
        "cv_mean"  : cv.mean(),
        "cv_std"   : cv.std(),
    }
    print(f"  {name:<28} {tr_acc:>10.4f} {te_acc:>10.4f} "
          f"{cv.mean():>10.4f} {cv.std():>8.4f}")

# ─────────────────────────────────────────────
# 5.  BEST MODEL — DETAILED REPORT
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["test_acc"])
best      = results[best_name]
best_model = best["model"]

print("\n" + "─" * 60)
print(f"  🏆  BEST MODEL  :  {best_name}")
print(f"      Test Accuracy : {best['test_acc']:.4f} ({best['test_acc']*100:.2f} %)")
print("─" * 60)

y_pred = best_model.predict(X_test_s)
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred,
                             target_names=le.classes_,
                             digits=4))

# ─────────────────────────────────────────────
# 6.  VISUALISATIONS — MODEL PERFORMANCE
# ─────────────────────────────────────────────

# 6-a  Accuracy bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Comparison — Accuracy", fontsize=14, fontweight="bold")

names      = list(results.keys())
test_accs  = [results[n]["test_acc"]  for n in names]
train_accs = [results[n]["train_acc"] for n in names]
cv_means   = [results[n]["cv_mean"]   for n in names]
cv_stds    = [results[n]["cv_std"]    for n in names]
short_names = [n.replace(" ", "\n") for n in names]

x = np.arange(len(names))
w = 0.35
ax = axes[0]
bars1 = ax.bar(x - w/2, train_accs, w, label="Train", color="#90CAF9", edgecolor="white")
bars2 = ax.bar(x + w/2, test_accs,  w, label="Test",  color="#4CAF50", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=8)
ax.set_ylim(0.8, 1.02)
ax.set_title("Train vs Test Accuracy")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(axis="y", alpha=0.3)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

ax = axes[1]
bars = ax.bar(x, cv_means, color="#FF9800", edgecolor="white")
ax.errorbar(x, cv_means, yerr=cv_stds, fmt="none",
            capsize=5, color="black", linewidth=1.5)
ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=8)
ax.set_ylim(0.8, 1.05)
ax.set_title("5-Fold Cross-Validation Accuracy")
ax.set_ylabel("CV Accuracy")
ax.grid(axis="y", alpha=0.3)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
fig.savefig(f"{OUT}/2_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# 6-b  Confusion matrices for ALL models
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes.flatten(), results.items()):
    pred = res["model"].predict(X_test_s)
    cm   = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Setosa", "Versicolor", "Virginica"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    acc_str = f"Acc={res['test_acc']*100:.1f}%"
    ax.set_title(f"{name}\n{acc_str}", fontsize=9)

plt.tight_layout()
fig.savefig(f"{OUT}/3_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# 6-c  Feature importance (Random Forest)
rf = results["Random Forest"]["model"]
importances = rf.feature_importances_
fi_sorted   = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(figsize=(8, 4))
names_fi  = [f.replace("Cm", "") for f, _ in fi_sorted]
values_fi = [v for _, v in fi_sorted]
colors_fi = ["#E91E63", "#9C27B0", "#3F51B5", "#009688"]
bars = ax.barh(names_fi, values_fi, color=colors_fi, edgecolor="white")
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importances — Random Forest", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, values_fi):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=10)
plt.tight_layout()
fig.savefig(f"{OUT}/4_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# 6-d  Decision Tree visualisation
dt = results["Decision Tree"]["model"]
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(dt, feature_names=[f.replace("Cm", "") for f in features],
          class_names=["Setosa", "Versicolor", "Virginica"],
          filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title("Decision Tree Visualisation (max_depth=4)", fontweight="bold", fontsize=13)
plt.tight_layout()
fig.savefig(f"{OUT}/5_decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()

# 6-e  Decision boundary (2 best petal features) for best model
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"Decision Boundary — {best_name} (Petal Features)",
             fontsize=13, fontweight="bold")

feature_pairs = [
    (2, 3, "Petal Length", "Petal Width"),
    (0, 2, "Sepal Length", "Petal Length"),
    (0, 1, "Sepal Length", "Sepal Width"),
]
cmap = plt.cm.get_cmap("Set2", 3)

for ax, (i, j, xl, yl) in zip(axes, feature_pairs):
    # re-train a fresh instance on 2 features
    sub_X   = X[:, [i, j]]
    from sklearn.preprocessing import StandardScaler as SS
    sc2 = SS()
    sub_Xs  = sc2.fit_transform(sub_X)   # fresh scaler for 2 features
    sub_Xtr, sub_Xte, ytr, yte = train_test_split(sub_Xs, y, test_size=0.2,
                                                    random_state=42, stratify=y)
    del sc2
    clf2 = type(best_model)(**{k: v for k, v in best_model.get_params().items()})
    clf2.fit(sub_Xtr, ytr)

    h  = 0.02
    x_min, x_max = sub_Xs[:, 0].min() - 0.5, sub_Xs[:, 0].max() + 0.5
    y_min, y_max = sub_Xs[:, 1].min() - 0.5, sub_Xs[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    for cls, col in zip(range(3), COLORS):
        idx = sub_Xs[y == cls]
        ax.scatter(idx[:, 0], idx[:, 1], c=col, s=40, edgecolors="black",
                   linewidth=0.5, label=le.classes_[cls].replace("Iris-", ""))
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(f"{xl} vs {yl}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

plt.tight_layout()
fig.savefig(f"{OUT}/6_decision_boundaries.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅  All plots saved.")

# ─────────────────────────────────────────────
# 7.  HYPERPARAMETER TUNING (Best Model)
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print(f"  🔧  HYPERPARAMETER TUNING — {best_name}")
print("─" * 60)

param_grids = {
    "Logistic Regression"    : {"C": [0.01, 0.1, 1, 10, 100]},
    "K-Nearest Neighbors"    : {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform","distance"]},
    "Decision Tree"          : {"max_depth": [2, 3, 4, 5, None], "criterion": ["gini","entropy"]},
    "Random Forest"          : {"n_estimators": [50, 100, 200], "max_depth": [3, 5, None]},
    "Support Vector Machine" : {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "Naive Bayes"            : {"var_smoothing": [1e-9, 1e-8, 1e-7]},
}

grid = GridSearchCV(
    best_model, param_grids[best_name],
    cv=5, scoring="accuracy", n_jobs=-1
)
grid.fit(X_train_s, y_train)
tuned_model = grid.best_estimator_
tuned_acc   = accuracy_score(y_test, tuned_model.predict(X_test_s))

print(f"  Best params : {grid.best_params_}")
print(f"  Tuned Acc   : {tuned_acc:.4f}  (before: {best['test_acc']:.4f})")

# ─────────────────────────────────────────────
# 8.  FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  📊  FINAL SUMMARY")
print("=" * 60)
summary = pd.DataFrame({
    "Model"      : list(results.keys()),
    "Train Acc"  : [f"{results[n]['train_acc']:.4f}" for n in results],
    "Test  Acc"  : [f"{results[n]['test_acc']:.4f}"  for n in results],
    "CV Mean"    : [f"{results[n]['cv_mean']:.4f}"   for n in results],
    "CV Std"     : [f"{results[n]['cv_std']:.4f}"    for n in results],
})
print(summary.to_string(index=False))
summary.to_csv(f"{OUT}/model_summary.csv", index=False)

# ─────────────────────────────────────────────
# 9.  LIVE PREDICTION DEMO
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  🌸  LIVE PREDICTION DEMO")
print("─" * 60)

sample_flowers = [
    [5.1, 3.5, 1.4, 0.2],    # typical setosa
    [6.0, 2.7, 5.1, 1.6],    # typical versicolor
    [6.9, 3.1, 5.4, 2.1],    # typical virginica
    [4.8, 3.0, 1.4, 0.3],    # setosa
    [7.7, 3.0, 6.1, 2.3],    # virginica
]

print(f"\n  Using best model: {best_name}")
print(f"  {'Sepal L':>8} {'Sepal W':>8} {'Petal L':>8} {'Petal W':>8}  →  Prediction")
print("  " + "-" * 55)
for fl in sample_flowers:
    scaled_fl = scaler.transform([fl])
    pred      = tuned_model.predict(scaled_fl)[0]
    species   = le.inverse_transform([pred])[0]
    prob      = tuned_model.predict_proba(scaled_fl)[0].max() * 100 if hasattr(tuned_model, "predict_proba") else None
    prob_str  = f"  ({prob:.1f}% confidence)" if prob else ""
    print(f"  {fl[0]:>8} {fl[1]:>8} {fl[2]:>8} {fl[3]:>8}  →  {species}{prob_str}")

print("\n" + "=" * 60)
print("  ✅  PROJECT COMPLETE — All outputs saved to:")
print(f"  {OUT}")
print("=" * 60)
