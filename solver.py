import torch
import torch.nn as nn

import torch.utils.data as data_utils
import torch.optim as optim

from my_dataset import Dataset
from data_processing import process_data
from logger import Logger

from model import LanguageNet, ConvLanguageNet


def test_loop(model, test_dataloader):

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        total = 0
        correct = 0
        running_loss = 0

        for i, data in enumerate(test_dataloader, 0):

            inputs, labels = data

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        print('Test loss: %.3f' % running_loss)
        print('Training acc: %.3f' % (correct / total * 100))

    return (correct / total * 100), running_loss


def training_loop(model, corpus, labels, num_epochs=100, train_split=0.9, model_name=None, device=torch.device("cpu")):

    # log_writer = Logger("./logs", model_name if model_name is not None else 'noname')

    train_split = int(train_split * len(corpus))
    train_dataset = Dataset(corpus[:train_split], labels[:train_split])
    test_dataset = Dataset(corpus[train_split:], labels[train_split:])

    trainloader = data_utils.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    testloader = data_utils.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

    print(f'%%%%%%%%%%%%%%%%%%%% Starting training of {model_name} %%%%%%%%%%%%%%%%%%%%')

    for epoch in range(30):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device=torch.device('cuda'))
            # print(inputs.shape, labels.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs.cpu(), labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if (epoch + 1) % 1 == 0:
            print('[%d] loss: %.3f' % (epoch + 1, running_loss))
            test_acc, test_loss = test_loop(model, testloader)
            # log_writer.scalar_summary("test_loss", test_loss.item(), epoch)
            # log_writer.scalar_summary("test_accuracy", test_acc, epoch)

        # log_writer.scalar_summary("train_loss", running_loss, epoch)
        running_loss = 0.0



    print('Finished Training')


def main():
    torch.manual_seed(24)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    data_dir = './clean_data/full'
    model_name = '3LayerModel'
    word_length = 8


    corpus, labels, vector_length, _, label_dict = process_data(data_dir, word_length=word_length)

    num_languages = len(label_dict.keys())

    for i in range(num_languages):
        count = len([l for l in labels if l == i])
        print(f'{label_dict[i]} word count = {count}')
    print(f'Vector length = {vector_length}')

    model = LanguageNet(vector_length, word_length, num_languages)
    model = model.to(device)
    # model = ConvLanguageNet(vector_length, word_length, num_languages)
    model.train()

    training_loop(model, corpus, labels, model_name='Linear', device=device)

    model = ConvLanguageNet(vector_length, word_length, num_languages)
    model = model.to(device)
    model.train()

    training_loop(model, corpus, labels, model_name='Convolution', device=device)


if __name__ == '__main__':
    main()
