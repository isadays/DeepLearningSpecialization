import xgboost as xgb
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
def compute_ks(y_true, y_pred):
    pred_pos = y_pred[y_true == 1]
    pred_neg = y_pred[y_true == 0]
    ks_stat, p_value = ks_2samp(pred_pos, pred_neg)
    return ks_stat, p_value

df = pd.read_csv("dataset.csv")  
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    random_state=42
)
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)[:, 1]

roc_auc_xgb = roc_auc_score(y_test, y_proba)
ks_xgb, p_val_xgb = compute_ks(y_test.values, y_proba)
print("XGBoost ROC AUC:", roc_auc_xgb)
print("XGBoost KS Statistic:", ks_xgb)
print("XGBoost KS p-value:", p_val_xgb)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

class TabTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=32, n_heads=4, n_layers=2, dropout=0.1):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        batch_size, n_features = x.shape
        x = x.unsqueeze(-1)  # Shape: (batch_size, n_features, 1)
        x_emb = self.embedding(x)  # Shape: (batch_size, n_features, embed_dim)
        x_trans = self.transformer_encoder(x_emb)
        x_pool = x_trans.mean(dim=1)  # Aggregate over features
        logits = self.fc(x_pool)
        return logits.squeeze(-1)

input_dim = X_train.shape[1]
model = TabTransformer(input_dim=input_dim, embed_dim=32, n_heads=4, n_layers=2, dropout=0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
epochs = 10   

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_tensor)
    loss = criterion(logits, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_logits = model(X_test_tensor)
    test_probas = torch.sigmoid(test_logits)
    test_probas_np = test_probas.cpu().numpy()

roc_auc_transformer = roc_auc_score(y_test, test_probas_np)
ks_transformer, p_val_transformer = compute_ks(y_test.values, test_probas_np)
print("Transformer-based model ROC AUC:", roc_auc_transformer)
print("Transformer-based model KS Statistic:", ks_transformer)
print("Transformer-based model KS p-value:", p_val_transformer)
