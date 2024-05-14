import torch
from torch.utils.data import DataLoader, Dataset
from my_mixkabrn_model.mixkabrn import MixKABRN

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(self.texts[idx], return_tensors="pt")

def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch['input_ids'].squeeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    # Example dataset and tokenizer
    texts = ["Hello world", "MixKABRN model training", "Sample text data"]
    tokenizer = lambda x: {'input_ids': torch.tensor([[ord(c) for c in x]])}

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = MixKABRN(input_dim=128, output_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    train_model(model, dataloader, optimizer, criterion, epochs=10)

