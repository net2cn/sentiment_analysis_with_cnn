# Meh, make it quick.
# I'm kinda sleepy now.
import os
import torch
from torchtext import data
from torchtext import datasets
import spacy
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
import tqdm

nlp = spacy.load('en')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True

def read_data(path, separator):
    # Kinda generic raw-data parser.
    # PhraseId SentenceId Phrase Sentiment
    # 0 1 2 3
    
    # The sentiment labels are:
    # 0 - negative
    # 1 - somewhat negative
    # 2 - neutral
    # 3 - somewhat positive
    # 4 - positive
    data = []

    with open(path, "r") as f:
        lines = f.read().splitlines()

    print("Processing raw-data...")

    if separator:
        for line in tqdm.tqdm(lines, ascii=True):
            data.append(line.split(separator))
    else:
        data = lines

    return data

# I would like to refactor this if possible.
# Nah, whatever, as long as it works.
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                dropout, pad_idx):
        super().__init__()
        # Word embeddings from input words.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Specify convs with filters of different sizes.
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels = 1, 
                          out_channels = n_filters, 
                          kernel_size = (fs, embedding_dim), padding=(fs - 1, 0)) for fs in filter_sizes
            ]
        )
        # Fully connected layer for final predictions.
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        # Drop nodes to increase robustness in training.
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)   # [batch size, sent len]
        embedded = self.embedding(text) # [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)    # [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs] # [batch size, n_filters, sent len - filter_sizes[n]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim = 1))  # [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)

def categorical_accuracy(preds, y):
    max_preds = preds.argmax(dim = 1, keepdim = True) # Get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm.tqdm(iterator):
        optimizer.zero_grad()
        predictions = model(batch.Pharse)
        loss = criterion(predictions, batch.Sentiment)
        acc = categorical_accuracy(predictions, batch.Sentiment)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.Pharse)
            loss = criterion(predictions, batch.Sentiment)
            acc = categorical_accuracy(predictions, batch.Sentiment)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def predict_class(model, sentence, sentences, min_len = 4):
    model.eval()
    tokenized = [t.text for t in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [sentences.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim = 1)
    return max_preds.item()

def main():
    # Prepare data.
    print("Preparing data...")
    sentence = data.Field(tokenize="spacy")
    sentiment = data.LabelField()

    fields = [(None, None), (None, None), ("Pharse", sentence), ('Sentiment', sentiment)]
    train_data, test_data = data.TabularDataset.splits(
                                            path = "datasets",
                                            train = "train.tsv",
                                            test = "test.tsv",
                                            format = "tsv",
                                            fields = fields,
                                            skip_header = True)
    # train_data, _ = train_data.split(split_ratio=0.05)    # Development setup, reduces data to shorten wait time.
    _, valid_data = train_data.split()                      # Submission setup, if you want train-validation, 
                                                            #   simply change "_" to "train_data"
    sentence.build_vocab(train_data,
                        vectors = "glove.6B.100d", 
                        unk_init = torch.Tensor.normal_)
    sentiment.build_vocab(train_data)

    print(vars(train_data[0]))
    print(vars(test_data[0]))
    print(sentiment.vocab.stoi)

    print("Setting up model...")
    # Parameters
    EPOCHES = 10
    BATCH_SIZE = 64
    INPUT_DIM = len(sentence.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [1, 2, 3, 4, 5]
    OUTPUT_DIM = len(sentiment.vocab)
    DROPOUT = 0.5
    PAD_IDX = sentence.vocab.stoi[sentence.pad_token]
    UNK_IDX = sentence.vocab.stoi[sentence.unk_token]

    # Model setup.
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    pretrained_embeddings = sentence.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    # If no pretrained model found
    if not os.path.exists("./model/model.pt"):
        print("No trained model detected! Training new...")
        if not os.path.exists("./model"):
            os.mkdir("model")
        
        best_valid_loss = float('inf')
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                                        (train_data, valid_data, test_data), 
                                                                        batch_size = BATCH_SIZE, 
                                                                        device = device,
                                                                        sort_key = lambda x: len(x.Pharse),
                                                                        sort_within_batch = False)

        print("Training...")

        for epoch in range(EPOCHES):
            print(f'Epoch: {epoch+1}')
            train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), './model/model.pt')
            
            print(f'\ttrain loss: {train_loss}\n\ttrain acc: {train_acc}')
            print(f'\t val. loss: {valid_loss}\n\t val. acc: {valid_acc}')

    # Otherwise we just do our predictions.
    print("Predicting...")
    model.load_state_dict(torch.load("./model/model.pt", map_location=device))
    model.eval()

    test_dataset = read_data("./datasets/test.tsv", "	")[1:]
    idx2label = {v: k for k, v in sentiment.vocab.stoi.items()}

    # MEH, FINALLY!
    csv = []
    csv.append("PhraseId,Sentiment")
    i = 0
    for line in tqdm.tqdm(test_dataset):
        pharse_id = line[0]
        prediction = idx2label[predict_class(model, line[2], sentence)]

        csv.append("{0},{1}".format(pharse_id, prediction))

        i += 1
    
    csv.append("")
    with open("submission.csv", "w") as outfile:
        outfile.write("\n".join(csv))               # After submission I'm gonna sleep 'till I die.
                                                    # I got a score of 0.61740, Yay.


if __name__ == "__main__":
    main()
