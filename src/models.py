import numpy as np

def stratified_k_fold_split(X, y, k=5):
    # Lấy index của từng lớp
    class0_idx = np.where(y == 0)[0] # Existing Customer
    class1_idx = np.where(y == 1)[0] # Attrited Customer
    
    # Xáo trộn độc lập từng lớp
    np.random.shuffle(class0_idx)
    np.random.shuffle(class1_idx)
    
    # Chia lớp 0 thành k fold
    folds_0 = np.array_split(class0_idx, k)
    folds_1 = np.array_split(class1_idx, k)

    folds = []
    for i in range(k):
        # Validation set = fold của lớp 0 + fold của lớp 1
        val_idx = np.concatenate([folds_0[i], folds_1[i]])
        
        # Training = toàn bộ còn lại
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        
        folds.append((train_idx, val_idx))
    
    return folds # (train_idx, val_idx)


def stratified_k_fold_cross_validation(X, y, k=5, learning_rate=0.1, num_iterations=2000):
    folds = stratified_k_fold_split(X, y, k=k)
    
    accuracies, precisions, recalls, f1s = [], [], [], []
    
    for i, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = LogisticRegressionNumPy(
            learning_rate=learning_rate,
            num_iterations=num_iterations
        )
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_val)
        acc, prec, rec, f1 = evaluate_metrics(y_val, predictions)
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
        print(f"Fold {i+1}: Acc={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    
    print("\n--- AVERAGE RESULTS ---")
    print(f"Accuracy : {np.mean(accuracies): .4f}")
    print(f"Precision: {np.mean(precisions): .4f}")
    print(f"Recall   : {np.mean(recalls): .4f}")
    print(f"F1-score : {np.mean(f1s): .4f}")
    
    return model, accuracies, precisions, recalls, f1s



def evaluate_metrics(y_true, y_pred):
    # y_true = 1 và y_pred = 1
    tp = np.sum((y_true == 1) & (y_pred == 1)) # True Positive

    # y_true = 0 và y_pred = 0
    tn = np.sum((y_true == 0) & (y_pred == 0)) # True Negative

    # y_true = 0 và y_pred = 1
    fp = np.sum((y_true == 0) & (y_pred == 1)) # False Positive

    # y_true = 1 và y_pred = 0
    fn = np.sum((y_true == 1) & (y_pred == 0)) # False Negative
    
    # 1. Accuracy: Tỷ lệ dự đoán đúng trên tổng số
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # 2. Precision: Trong số dự đoán là khách hàng rời đi, bao nhiêu thực sự là rời đi? 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # 3. Recall (Sensitivity): Trong số khách hàng thực sự Churn, tìm ra được bao nhiêu?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 4. F1-Score: Trung bình điều hòa của Precision và Recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

class LogisticRegressionNumPy:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.lr = learning_rate
        self.n_iters = num_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    # Hàm Sigmoid: Chuyển đổi z thành xác suất [0, 1]
    def _sigmoid(self, z):
        # Tránh tràn số (overflow)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # Hàm Mất mát (Binary Cross Entropy / Log Loss)
    def _compute_loss(self, y_true, y_pred):
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    def fit(self, X, y):
        # Khởi tạo tham số
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # 1. Forward pass (Tính toán dự đoán)
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # 2. Tính Loss (để theo dõi)
            loss = self._compute_loss(y, y_predicted)
            self.losses.append(loss)

            # 3. Backward pass (Tính Gradient)
            # Đạo hàm của hàm mất mát theo w và b
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 4. Cập nhật tham số (Optimization Step)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # Hàm đoán xác suất
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    # Hàm dự đoán nhãn
    def predict(self, X, threshold=0.3):
        y_predicted_cls = self.predict_proba(X)
        return np.array([1 if i > threshold else 0 for i in y_predicted_cls])