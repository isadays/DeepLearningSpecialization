import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd

# Transformer Architecture
class PropensityTransformer(nn.Module):
    def __init__(self, num_categorical_feats, num_continuous_feats, embedding_dims, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(PropensityTransformer, self).__init__()
        
        # Embedding for categorical variables
        self.embeddings = nn.ModuleList([nn.Embedding(cat_dim, emb_dim) for cat_dim, emb_dim in embedding_dims])
        emb_dim_total = sum([emb_dim for _, emb_dim in embedding_dims])

        self.input_dim = emb_dim_total + num_continuous_feats

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_cat, x_cont):
        emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        emb_cat = torch.cat(emb, dim=1)
        x = torch.cat([emb_cat, x_cont], dim=1).unsqueeze(1)  # Add seq dim

        x_transformed = self.transformer_encoder(x)
        x_transformed = x_transformed.squeeze(1)
        out = self.fc(x_transformed)
        return out

# Custom Dataset
class CustomerDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, target_col):
        self.df = df
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cat_feats = torch.tensor(row[self.cat_cols].values, dtype=torch.long)
        cont_feats = torch.tensor(row[self.cont_cols].values, dtype=torch.float)
        target = torch.tensor(row[self.target_col], dtype=torch.float)
        return cat_feats, cont_feats, target

# Training Function
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_cat, x_cont, y in dataloader:
            optimizer.zero_grad()
            preds = model(x_cat, x_cont).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")

# Monthly ROC and KS evaluation function
def evaluate_by_month(model, df, date_col, cat_cols, cont_cols, target_col):
    model.eval()
    results = []

    df['month'] = pd.to_datetime(df[date_col]).dt.to_period('M')

    for month, group in df.groupby('month'):
        dataset = CustomerDataset(group, cat_cols, cont_cols, target_col)
        dataloader = DataLoader(dataset, batch_size=1024)

        all_preds, all_targets = [], []

        with torch.no_grad():
            for x_cat, x_cont, y in dataloader:
                preds = model(x_cat, x_cont).squeeze().cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.cpu().numpy())

        roc_auc = roc_auc_score(all_targets, all_preds)

        fpr, tpr, _ = roc_curve(all_targets, all_preds)
        ks_stat = np.max(tpr - fpr)

        results.append({
            'Month': str(month),
            'ROC_AUC': roc_auc,
            'KS': ks_stat
        })

    return pd.DataFrame(results)

# Example Usage
# df = your_dataframe_here
# cat_cols = ['age_category', 'debit_category']
# cont_cols = ['account_balance', 'continuous_var']
# target_col = 'combined_target'
# date_col = 'interaction_date'

# embedding_dims = [(df[col].nunique(), min(50, (df[col].nunique()+1)//2)) for col in cat_cols]

# model = PropensityTransformer(
#     num_categorical_feats=len(cat_cols),
#     num_continuous_feats=len(cont_cols),
#     embedding_dims=embedding_dims,
#     hidden_dim=128,
#     num_heads=4,
#     num_layers=2,
#     dropout=0.1
# )

# dataset = CustomerDataset(df, cat_cols, cont_cols, target_col)
# dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# train_model(model, dataloader, criterion, optimizer, epochs=5)

# performance_by_month = evaluate_by_month(model, df, date_col, cat_cols, cont_cols, target_col)
# print(performance_by_month)


# Full Transformer Architecture (Encoder-Decoder) for Sequence-to-Sequence tasks
class FullTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout=0.1, output_dim=1):
        super(FullTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu'
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, src, tgt):
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        output = output.mean(dim=1)  # Aggregate sequence dimension
        out = self.fc_out(output)
        return out
