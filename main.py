import seaborn as sns
import matplotlib.pyplot as plt
import torch


from BigramLanguageModel import BigramLanguageModel


# open the file and read it into a variable called text
def load_text():
    with open('./input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def build_encoder(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s : [stoi[c] for c in s]
    decode = lambda c : ''.join(itos[i] for i in c)

    return encode, decode, len(chars), itos, stoi


def get_batch(split, batch_size = 4, block_size = 8):
    data = training_data if split == 'train' else validation_data

    randomx = torch.randint(len(data) - block_size, (batch_size,) )
    x = torch.stack([data[i: i + block_size] for i in randomx])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in randomx])

    return x,y

def plot_bigram_heatmap(model, itos):
    """
    Plots a heatmap of bigram logits.
    X-axis: next character
    Y-axis: current character
    """
    with torch.no_grad():
        logits = model.token_embedding.weight.detach().cpu().numpy()

    chars = [itos[i] for i in range(len(itos))]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        logits,
        xticklabels=chars,
        yticklabels=chars,
    )
    plt.xlabel("Next character")
    plt.ylabel("Current character")
    plt.title("Bigram Logits Heatmap")
    plt.tight_layout()
    plt.show()


def plot_bigram_probability_heatmap(model, itos):
    with torch.no_grad():
        logits = model.token_embedding.weight
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    chars = [itos[i] for i in range(len(itos))]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        probs,
        xticklabels=chars,
        yticklabels=chars,
        cmap="magma"
    )
    plt.xlabel("Next character")
    plt.ylabel("Current character")
    plt.title("Bigram Transition Probabilities")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    loss_history = []
    # data
    text = load_text()
    print("Raw text length:", len(text))
    encode, decode, vocabulary_size, itos, stoi  = build_encoder(text)
    print("Vocabulary size:", vocabulary_size)

    encoded_text = encode(text)

    data = torch.tensor(encode(text), dtype=torch.long) #storing data in tensor
    print("data tensor" ,data)
    print("data shape", data.shape)

    number_of_training_data = int(0.9 * data.size(0))
    training_data, validation_data = data[:number_of_training_data], data[number_of_training_data:]

    print("training data shape", training_data.shape, "test data shape", validation_data.shape)


    # model
    blm = BigramLanguageModel(vocabulary_size)
    # optimizer
    optimizer = torch.optim.AdamW(blm.parameters(), lr= 1e-3)
    # training
    for step in range(100000):
        x_training_batch, y_training_batch = get_batch('train')
        logits, loss = blm(x_training_batch, y_training_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.show()

    print("loss" , loss.item())
    print("logits shape", logits.shape) # batch size * vocab_size
    print(decode(blm.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
    plot_bigram_heatmap(blm, itos)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
