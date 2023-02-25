from copy import deepcopy

import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report


class Trainer():

    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.best_loss = np.inf

    def update_best_model(self, model, loss):
        if loss <= self.best_loss:
            self.best_loss = loss 
            self.best_model = deepcopy(model.state_dict()) # Update best model weights.

    def train(self, model,
              loss_func,
              optimizer,
              scheduler,
              train_loader,
              valid_loader,
              index_to_tag,
              device):

        for epoch in range(self.config.n_epochs):

            model.train()
            total_tr_loss = 0
            for step, mini_batch in enumerate(tqdm(train_loader)):
                input_ids, multilabels = mini_batch['input_ids'], mini_batch['multilabels']
                input_ids = input_ids.to(device)
                multilabels = multilabels.permute(1, 0).to(device)
                attention_mask = mini_batch['attention_mask']
                attention_mask = attention_mask.to(device)
                batch_size = input_ids.shape[0]

                # You have to reset the gradients of all model parameters
                # before to take another step in gradient descent.
                optimizer.zero_grad()

                output = model(input_ids, 
                               attention_mask=attention_mask)

                loss = 0.0
                for j in range(len(output)):
                    loss += loss_func(output[j], 
                                      multilabels[j])
                loss = loss / batch_size

                loss.backward()
                # track train loss
                total_tr_loss += loss.item()
                # update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()


            # Calculate the average loss over the training data.
            avg_tr_loss = total_tr_loss / len(train_loader)
            # loss={:.4e}
            print('Epoch {} - loss={:.4f}'.format(
                epoch+1, avg_tr_loss
            ))
            
            model.eval()
            total_val_loss = 0
            y_pred = [[] for _ in range(len(index_to_tag))]
            y_true = [[] for _ in range(len(index_to_tag))]

            with torch.no_grad():
                for step, mini_batch in enumerate(tqdm(valid_loader)):
                    input_ids, multilabels = mini_batch['input_ids'], mini_batch['multilabels']
                    input_ids = input_ids.to(device)
                    multilabels = multilabels.permute(1, 0).to(device)
                    attention_mask = mini_batch['attention_mask']
                    attention_mask = attention_mask.to(device)

                    output = model(input_ids,
                                   attention_mask=attention_mask)

                    loss = 0.0
                    for j in range(len(output)):
                        loss += loss_func(output[j], 
                                          multilabels[j])

                        output[j] = output[j].to('cpu')
                        curr_y_pred = output[j].argmax(dim=1)
                        y_pred[j].append(curr_y_pred)
                        y_true[j].append(multilabels[j].to('cpu'))

            loss = loss / batch_size
            total_val_loss += loss.item()

            for j in range(len(y_true)):
                y_true[j] = torch.cat(y_true[j], dim=0)
                y_pred[j] = torch.cat(y_pred[j], dim=0)

            y_pred = [t.tolist() for t in y_pred]
            y_true = [t.tolist() for t in y_true]
            # y_pred = y_true = (num_of_tag, batch_size)

            y_true = np.array(y_true).T
            y_pred = np.array(y_pred).T
            # y_pred = y_true = (batch_size, num_of_tag)

            avg_val_loss = total_val_loss / len(valid_loader)
            avg_val_acc = accuracy_score(y_true, y_pred)
            avg_val_micro_f1 = f1_score(y_true, y_pred, average='micro')
            avg_val_macro_f1 = f1_score(y_true, y_pred, average='macro')

            self.update_best_model(model, avg_val_loss)

            print('Validation - loss={:.4f} acc={:.4f} micro_f1={:.4f} macro_f1={:.4f} best_loss={:.4f}'.format(
                avg_val_loss, avg_val_acc, avg_val_micro_f1, avg_val_macro_f1, self.best_loss
            ))

        model.load_state_dict(self.best_model)
        return model
        
    def test(self, 
             model,
             test_loader,
             index_to_tag,
             device):

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss and accuracy for this epoch.
        y_pred = [[] for _ in range(len(index_to_tag))]
        y_true = [[] for _ in range(len(index_to_tag))]

        with torch.no_grad():
            for step, mini_batch in enumerate(tqdm(test_loader)):
                input_ids, multilabels = mini_batch['input_ids'], mini_batch['multilabels']
                input_ids = input_ids.to(device)
                multilabels = multilabels.permute(1, 0).to(device)
                attention_mask = mini_batch['attention_mask']
                attention_mask = attention_mask.to(device)

                output = model(input_ids,
                               attention_mask=attention_mask)

                for j in range(len(output)):
                    output[j] = output[j].to('cpu')
                    curr_y_pred = output[j].argmax(dim=1)
                    y_pred[j].append(curr_y_pred)
                    y_true[j].append(multilabels[j].to('cpu'))

        for j in range(len(y_true)):
            y_true[j] = torch.cat(y_true[j], dim=0)
            y_pred[j] = torch.cat(y_pred[j], dim=0)

        y_pred = [t.tolist() for t in y_pred]
        y_true = [t.tolist() for t in y_true]

        y_true = np.array(y_true).T
        y_pred = np.array(y_pred).T
        
        avg_test_acc = accuracy_score(y_true, y_pred)
        avg_test_micro_f1 = f1_score(y_true, y_pred, average='micro')
        avg_test_macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        print('Test - acc={:.4f} micro_f1={:.4f} macro_f1={:.4f}'.format(
            avg_test_acc, avg_test_micro_f1, avg_test_macro_f1
        ))

        print(classification_report(y_true, y_pred))