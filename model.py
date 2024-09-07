import torch
from torch import nn
from transformers import BertTokenizer, BertModel

class TradingModel(nn.Module):
    def __init__(self):
        super(TradingModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 1)  # Assuming binary classification: trade or not trade

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids, attention_mask, return_dict=False)
        return self.fc(pooled_output)

    def predict(self, message):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(message, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
        return torch.sigmoid(output).item() > 0.5  # Returns True if message is trade-related
    
from torch.utils.data import DataLoader, Dataset

class MessageDataset(Dataset):
    def __init__(self, data_file):
        self.data = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        with open(data_file, 'r') as f:
            for line in f:
                label, text = line.strip().split("\t")
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
                self.data.append((int(label), inputs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, inputs = self.data[idx]
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), torch.tensor(label)

def train_model():
    dataset = MessageDataset('dataset.txt')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = TradingModel()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(3):
        for input_ids, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train_model()

