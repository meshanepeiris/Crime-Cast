import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
# from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler
import importlib_metadata
from tqdm import tqdm
from evaluation import evaluate

# load the csv to fine tune

headlines = pd.read_csv('/../../csv/headlines_fine_tune.csv', encoding='ISO-8859-1')

labels = headlines['class'].values

possible_labels = {'positive', 'negative', 'neutral'}
label_encoder = LabelEncoder()
labels_numeric = label_encoder.fit_transform(labels)
# positive = 2, neutral = 1, negative = 0

# x = headlines['headline']
# y = headlines['class']
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Use following code to find the max length of the data set
# max_length = 0
#
# for headline in headlines['headline']:
#     input_ids = tokenizer.encode(headline, add_special_tokens=True)
#     max_length = max(max_length, len(input_ids))
#
# print('Max sentence length: ', max_length)

# max sentence length is 150


input_ids = []
attention_masks = []

# tokenize each headline in the training set
for headline in headlines['headline']:
     encoded_dict = tokenizer.encode_plus(
          headline, # headline to encode
          add_special_tokens=True,  # add beg and end tokens
          max_length=150,  # pad shorter headlines
          padding='max_length',
          return_attention_mask=True,  # attention mask for transformer
          return_tensors='pt',  # return the tensor
          truncation=True
     )
     input_ids.append(encoded_dict['input_ids'])
     attention_masks.append(encoded_dict['attention_mask'])

# list -> tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels_tensor = torch.tensor(labels_numeric, dtype=torch.long)

# print(type(input_ids))
# print(type(attention_masks))
# print(type(labels_tensor))

dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
# split the dataset into training and validation (test)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# split randomly
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# print('{:>5,} training samples'.format(train_size))
# print('{:>5,} validation samples'.format(val_size))

# set up dataloaders for training loop
batch_size = 32
train_dataloader = DataLoader(
            train_dataset,  # the training samples.
            sampler = RandomSampler(train_dataset), # select batches randomly
            batch_size = batch_size # trains with this batch size.
        )

# test dataloader setup, for validation
test_dataloader = DataLoader(
            val_dataset,
            batch_size = batch_size
        )

# now we get into training
pretrained_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 3,
    output_attentions = False,
    output_hidden_states = False,
)

# use AdamW optimizer
optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=.00002, eps=1e-8)
pretrained_model.to('cuda')
pretrained_model.train()

for epoch in range(5):  # Number of epochs
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        # Move batch to GPU if available
        input_ids = batch[0].to('cuda')
        attention_masks = batch[1].to('cuda')
        labels = batch[2].to('cuda')

        # zero gradients
        optimizer.zero_grad()

        # forward pass
        outputs = pretrained_model(
            input_ids=input_ids.to('cuda'),
            attention_mask=attention_masks.to('cuda'),
            labels=labels.to('cuda')
        )

        # compute loss
        loss = outputs.loss
        total_loss += loss.item()

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Average Training Loss for Epoch {epoch + 1}: {avg_loss}")

def evaluate(model, dataloader):
    model.eval()  # Set model to evaluation mode
    total_eval_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to('cuda')
            attention_masks = batch[1].to('cuda')
            labels = batch[2].to('cuda')

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=labels
            )

            loss = outputs.loss
            total_eval_loss += loss.item()

            # Get predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(predictions == labels).item()

    avg_eval_loss = total_eval_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)
    print(f"Validation Loss: {avg_eval_loss}")
    print(f"Validation Accuracy: {accuracy}")

# Evaluate the model
evaluate(pretrained_model, test_dataloader)

model_save_path = "./bert_finetuned.pt"
torch.save(pretrained_model.state_dict(), model_save_path)

# Save the tokenizer
tokenizer_save_path = "./bert_tokenizer/"
tokenizer.save_pretrained(tokenizer_save_path)

# Evaluate the model
evaluate(pretrained_model, test_dataloader)

model_save_path = "./bert_finetuned.pt"
torch.save(pretrained_model.state_dict(), model_save_path)

# Save the tokenizer
tokenizer_save_path = "./bert_tokenizer/"
tokenizer.save_pretrained(tokenizer_save_path)