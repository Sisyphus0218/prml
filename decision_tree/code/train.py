import joblib
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import data_setup


train_dir = "data"
test_dir = "data"

X_train, y_train, X_test, y_test, class_names = data_setup.create_datasets(
    train_dir=train_dir, test_dir=test_dir
)

start_time = time.time()

clf = DecisionTreeClassifier(max_depth=15, random_state=42)
print("start training...")
clf.fit(X_train, y_train)

total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} seconds")

joblib.dump(clf, "models/cifar10_decision_tree.pkl")

y_pred = clf.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average="macro")
test_recall = recall_score(y_test, y_pred, average="macro")
test_f1 = f1_score(y_test, y_pred, average="macro")

print(
    f"test_acc : {test_acc:.4f} | "
    f"test_precision : {test_precision:.4f} | "
    f"test_recall : {test_recall:.4f} | "
    f"test_f1 : {test_f1:.4f}"
)
