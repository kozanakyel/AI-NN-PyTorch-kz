from io import open
import glob
import os
import unicodedata
import string

import torch
import torch.nn as nn

# DATA_PATH = 'my_research/data/'
# all_letters = string.ascii_letters + " .,;'"
# n_letters = len(all_letters)


class UtilityService:
    def __init__(self):
        self.DATA_PATH = 'my_research/data/'
        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters)
    
    @staticmethod
    def findFiles(path):
        return glob.glob(path)

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    @staticmethod
    def unicodeToAscii(s):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn" and c in all_letters
    )

    # Read a file and split into lines
    @staticmethod
    def readLines(filename):
        lines = open(filename, encoding="utf-8").read().strip().split("\n")
        return [UtilityService.unicodeToAscii(line) for line in lines]


class CategoryService:
    # Build the category_lines dictionary, a list of names per language
    def __init__(self):
        self._utility_service = UtilityService()
        self.category_lines = {}
        self.all_categories = []
        self._create_category_and_lines()

    def _create_category_and_lines(self):
        for filename in self._utility_service.findFiles(f"{self._utility_service.DATA_PATH}names/*.txt"):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self._utility_service.readLines(filename)
            self.category_lines[category] = lines

    # n_categories = len(all_categories)



###########   PART 2   ################

class TorchLetterUtilityService:
    # Find letter index from all_letters, e.g. "a" = 0
    @staticmethod
    def letterToIndex(letter):
        return UtilityService.all_letters.find(letter)

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    @staticmethod
    def letterToTensor(letter):
        tensor = torch.zeros(1, UtilityService.n_letters)
        tensor[0][TorchLetterUtilityService.letterToIndex(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    @staticmethod
    def lineToTensor(line):
        tensor = torch.zeros(len(line), 1, UtilityService.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][TorchLetterUtilityService.letterToIndex(letter)] = 1
        return tensor


#############      PART 3         #############



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

category_service = CategoryService()
utils = UtilityService()
n_categories = len(category_service.all_categories)
n_hidden = 128
rnn = RNN(utils.n_letters, n_hidden, n_categories)


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


###########     PART 4       #########

import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


##############   PART 5     #########


criterion = nn.NLLLoss()
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


##############   PART 6    #########

import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


if __name__ == '__main__':
    # print(findFiles(f"{DATA_PATH}names/*.txt"))
    # print(unicodeToAscii("Ślusàrski"))
    # print(f'All letters: {all_letters}, letters length: {n_letters}')   # 57 n_letters
    # print(all_categories, category_lines['Polish'])
    # print(category_lines['Italian'][:5])
    # print(letterToTensor('J'), letterToTensor('K').shape)
    # print(lineToTensor('Jones').size())
    # print(n_categories)       #  18
    # input = lineToTensor('Albert')
    # hidden = torch.zeros(1, n_hidden)

    # output, next_hidden = rnn(input[0], hidden)
    # print(output)
    # print(categoryFromOutput(output))
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print('category =', category, '/ line =', line)
    
    
    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print ``iter`` number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output, category_service.all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0   
    
    # PLOT TRAIN VALIDATION LOSS      
    plt.figure()
    plt.plot(all_losses)
    plt.savefig(f'{DATA_PATH}val_train_loss.png')
    
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Just return an output given a line
    def evaluate(line_tensor):
        hidden = rnn.initHidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        return output

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.savefig(f'{DATA_PATH}heatmap.png')
    
    predict('Dovesky')
    predict('Jackson')
    predict('Satoshi')