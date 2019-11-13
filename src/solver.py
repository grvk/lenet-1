import torch
import numpy as np

class Solver(object):

    BACKUP_PATH_TEMPLATE = './backups/{lr}_{i}_{total_i}.pth'

    def __init__(self, model, train_data, val_data, **options):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

        # add check
        self.train_loader_options = options["train_data_loader"]
        self.val_loader_options = options["val_data_loader"]
        self.back_up_model_every = options["back_up_model_every"] or 0
        self.epochs = options["epochs"] or 1

        self.train_loss_history = []
        self.val_loss_history = []

    def _get_val_set_loss(self):
        is_model_training = self.model.training
        self.model.eval()

        val_loader = torch.utils.data.DataLoader( \
            dataset = self.val_data, **self.val_loader_options)

        val_loss_total = 0
        val_batches = 0
        with torch.no_grad():
            for (inputs, labels) in val_loader:
                out = self.model(inputs)
                loss = self.model.criterion(out, labels).item()
                val_loss_total += loss
                val_batches += 1

        self.model.train(is_model_training)
        return val_loss_total / val_batches

    def _get_train_set_accuracy(self):
        is_model_training = self.model.training
        self.model.eval()

        train_loader = torch.utils.data.DataLoader( \
            dataset = self.train_data, **self.train_loader_options)
        train_acc = -1

        with torch.no_grad():
            correct = 0
            total = 0

            for (inputs, labels) in train_loader:
                out = self.model(inputs)
                predicted = torch.argmax(out.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total

        self.model.train(is_model_training)
        return train_acc

    def _get_val_set_accuracy(self):
        is_model_training = self.model.training
        self.model.eval()

        val_loader = torch.utils.data.DataLoader( \
            dataset = self.val_data, **self.val_loader_options)
        val_acc = -1

        with torch.no_grad():
            correct = 0
            total = 0
            for (inputs, labels) in val_loader:
                out = self.model(inputs)
                predicted = torch.argmax(out.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            val_acc = correct / total

        self.model.train(is_model_training)
        return val_acc

    def _get_learning_rate(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def _save_model_params(self, cur_epoch, learning_rate, optimizer):

        state = {
            'model_state': self.model.state_dict(),
            'epoch': cur_epoch,
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history,
            'optimizer_state': optimizer.state_dict(),
            'learning_rate': learning_rate
        }

        torch.save(state, \
            Solver.BACKUP_PATH_TEMPLATE.format( \
            lr=learning_rate, i=cur_epoch, total_i=cur_epoch+self.epochs))

    def train(self, optimizer, start_i=0):

        print("Training for {} epochs".format(self.epochs))
        self.model.train()

        every_i = self.back_up_model_every
        lr = self._get_learning_rate(optimizer)

        for epoch_i in np.arange(start_i, start_i + self.epochs):

            train_loader = torch.utils.data.DataLoader( \
                dataset = self.train_data, **self.train_loader_options)

            train_loss = 0
            batch_num = 0
            for i, data in enumerate(train_loader):
                inputs, labels = data

                optimizer.zero_grad()

                out = self.model(inputs)
                loss = self.model.criterion(out, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                batch_num += 1

            train_loss /= batch_num
            val_loss = self._get_val_set_loss()

            print("Epoch[{epoch}]  train_loss={tl}  val_loss={vl}".format( \
                epoch=epoch_i, tl=train_loss, vl=val_loss))

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            if every_i and ((epoch_i + 1) % every_i == 0):
                self._save_model_params(epoch_i + 1, lr, optimizer)

        train_acc = self._get_train_set_accuracy()
        val_acc = self._get_val_set_accuracy()

        self._save_model_params(start_i + self.epochs, lr, optimizer)
        print("Done training. Train Accuracy = {ta}. Val accuracy = {va}".
            format(ta = train_acc, va = val_acc))
