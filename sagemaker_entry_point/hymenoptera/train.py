import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import create_dataloaders
from model import load_model_resnet18


def _one_epoch(model, dataloader, criterion, optimizer, device):
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def train(args):
    # 1. Configuração de Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")

    dataloaders = create_dataloaders(args.data_dir, args.batch_size)

    model = load_model_resnet18()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

    print("Iniciando treinamento...")
    for epoch in range(args.epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss, epoch_acc = _one_epoch(model, dataloaders[phase], criterion, optimizer, device)

            print(f'Epoch {epoch+1}/{args.epochs} | {phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    save_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Modelo salvo em: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hiperparâmetros enviados pelo Notebook
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)

    # Variáveis de Ambiente do SageMaker Container
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))

    args = parser.parse_args()
    train(args)