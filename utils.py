import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

@torch.no_grad()
def generate(model, device, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    idx = idx.to(device) # Ensuring idx is on the same device as the model
    
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
    

@torch.inference_mode()
def evaluate(model, device, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss



# Dataset preparation
class LogEventDataset(Dataset):
    def __init__(self, log_events, tokenizer, max_log_event_length):
        print("LogEventDataset init", len(log_events))
        self.log_events = log_events
        self.tokenizer = tokenizer
        self.max_log_event_length = max_log_event_length

    def __len__(self):
        return len(self.log_events)

    def get_output_length(self):
        return self.max_log_event_length + 1 # <START> token followed by words

    def encode(self, log_event):
        #return self.tokenizer.encode(log_event).ids
        return self.tokenizer.EncodeAsIds(log_event)

    def decode(self, token_ids):
        #return self.tokenizer.decode(token_ids)
        return self.tokenizer.DecodeIds(token_ids)

    def __getitem__(self, idx):
        log_event = self.log_events[idx]
        token_ids = self.encode(log_event)
        # Adjusting for [CLS] at the beginning and [SEP] at the end, within the max length limit
        token_ids = [1] + token_ids[:self.max_log_event_length - 2] + [2]  # -2 accounts for [CLS] and [SEP]

        x = torch.zeros(self.max_log_event_length, dtype=torch.long)
        y = torch.zeros(self.max_log_event_length, dtype=torch.long)
        
        x[:len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        # Shifted by one for language modeling, starting after [CLS], ending before [SEP]
        y[:len(token_ids)-1] = torch.tensor(token_ids[1:], dtype=torch.long)
        
        # Masking the loss at the inactive locations, considering [SEP] is the last meaningful token
        y[len(token_ids)-1:] = -1
        return x, y

def create_datasets(input_file, tokenizer, max_log_event_length):

    # preprocessing of the input text file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    log_events = [line.strip() for line in lines if line.strip()] # get rid of any empty strings

    # partition the input data into a training and the test set
    test_set_size = min(2000, int(len(log_events) * 0.1)) # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(log_events)).tolist()
    train_log_events = [log_events[i] for i in rp[:-test_set_size]]
    test_log_events = [log_events[i] for i in rp[-test_set_size:]]
    print(f"Training set size: {len(train_log_events)}")
    print(f"Test set size: {len(test_log_events)}")

    # wrap in dataset objects
    train_dataset = LogEventDataset(train_log_events, tokenizer, max_log_event_length)
    test_dataset = LogEventDataset(test_log_events, tokenizer, max_log_event_length)

    return train_dataset, test_dataset


class InfiniteDataLoader:
    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch