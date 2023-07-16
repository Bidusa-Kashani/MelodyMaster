import torch
import torch.nn as nn

MAX_LENGTH = 512


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, device="cpu"):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class ConditionedAttnGRUDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, alephbert_tokenizer, alephbert_embeddings=None, lyricist_tfidf=None,
                 device="cpu"):
        super(ConditionedAttnGRUDecoder, self).__init__()
        self.output_size = output_size
        self.optimizer = None
        self.alephbert_tokenizer = alephbert_tokenizer
        self.embedding = alephbert_embeddings
        self.attention = BahdanauAttention(hidden_size, device=device)
        self.lyricist_tfidf = lyricist_tfidf
        self.lyricist_embedding = nn.Linear(lyricist_tfidf.shape[1], hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, input_lyricist, input_seq, target_tensor=None):
        decoder_input = input_seq[:, 0].unsqueeze(1)
        decoder_hidden = self.lyricist_embedding(nn.functional.one_hot(input_lyricist,
                                                                       num_classes=self.output_size).float() @ self.lyricist_tfidf).unsqueeze(
            0)
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = torch.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        keys = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(embedded, keys)
        context = context.permute(1, 0, 2)
        output, hidden = self.gru(embedded, context)
        output = self.out(output)
        output = torch.log_softmax(output, dim=-1)
        return output, hidden, attn_weights

    def train_all(self, train_dataloader, val_dataloader, optimizer, compute_metrics, epochs=10):
        self.optimizer = optimizer
        self.train()
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for batch in train_dataloader:
                input_lyricist = batch["labels"].to(self.device)
                target_seq = batch["input_ids"].to(self.device)
                decoder_outputs, decoder_hidden, attentions = self.forward(input_lyricist=input_lyricist,
                                                                           input_seq=target_seq,
                                                                           target_tensor=torch.cat(
                                                                               [target_seq[:, 1:], torch.zeros(
                                                                                   target_seq.shape[0], 1).long().to(
                                                                                   self.device)], dim=-1))
                loss = loss_fn(decoder_outputs, target_seq[:, 1:])
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    input_lyricist = batch["labels"].to(self.device)
                    target_seq = batch["input_ids"].to(self.device)
                    decoder_outputs, decoder_hidden, attentions = self.forward(input_lyricist=input_lyricist,
                                                                               input_seq=target_seq,
                                                                               target_tensor=target_seq[:, 1:])
                    print(decoder_outputs.shape, target_seq[:, 1:].shape)
                    loss = loss_fn(decoder_outputs, target_seq[:, 1:])
                    metrics = compute_metrics(decoder_outputs, target_seq[:, 1:])
                    print(f"Validation loss: {loss}, metrics: {metrics}")
