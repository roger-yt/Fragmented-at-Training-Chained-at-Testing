import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from utils.utils import do_test

Tokenizer = AutoTokenizer.from_pretrained("gpt2")
Tokenizer.pad_token = Tokenizer.eos_token

class MLPTrainer():
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.model_device = next(self.model.parameters()).device

    def train(self, train_loader, eval_loader, num_epochs, save_path=None, if_print=True, print_every=1, test_staff=None, save_every=2):
        if (save_path is not None) and (not os.path.exists(save_path)):
            os.makedirs(save_path)
        eval_acc, eval_loss = self.evaluate(eval_loader)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch in tqdm(train_loader):
                # print("hi")
                inputs = batch["input"].to(self.model_device)
                labels = batch["label"].to(self.model_device).squeeze()
                # print("labels=", Tokenizer.decode(labels[0], skip_special_tokens=True))
                outs = self.model(inputs)
                # print("outs=", outs.size())
                # print("labels=", labels.size())
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(outs, labels)
                # print("loss=", loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            eval_acc, eval_loss = self.evaluate(eval_loader)
            if if_print and (epoch+1) % print_every == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Eval Accuracy: {eval_acc}, Eval Loss: {eval_loss}")
                print("test_staff=", test_staff)
            if (save_path  is not None) and (epoch+1) % save_every==0:
                torch.save({"epoch": epoch, 
                            "train_loss": running_loss/len(train_loader), 
                            "eval_loss": eval_loss,
                            "optimizer_state_dict": self.optimizer.state_dict(), 
                            "model_state_dict":self.model.state_dict()}
                           , f"{save_path}/checkpoint-{epoch}.pt")
        return eval_acc

    def evaluate(self, eval_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_loader:
                inputs = batch["input"].to(self.model_device).squeeze()
                labels = batch["label"].to(self.model_device).squeeze()
                outputs = self.model(inputs)
                loss_fct = CrossEntropyLoss()
                test_loss = loss_fct(outputs, labels)
                # print("outputs=", outputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.squeeze()
                # print("predicted=", Tokenizer.batch_decode(predicted))
                # print("labels=", Tokenizer.batch_decode(labels))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        return acc, test_loss.item()
    

class ProbeTrainer():
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.model_device = next(self.model.parameters()).device

    def train(self, train_loader, eval_loader, num_epochs, save_path=None, if_print=True, print_every=1, test_staff=None):
        if (save_path is not None) and (not os.path.exists(save_path)):
            os.makedirs(save_path)
        eval_acc, eval_loss = self.evaluate(eval_loader)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch in train_loader:
                # print("hi")
                inputs = batch["input"].to(self.model_device)
                labels = batch["label"].to(self.model_device).squeeze()
                # ids = batch["ids"].to(self.model_device)
                # print("ids=", Tokenizer.decode(ids[0]))
                # print("labels=", Tokenizer.decode(labels[0], skip_special_tokens=True))
                outs = self.model(inputs)
                # print("outs=", outs.size())
                # print("labels=", labels.size())
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(outs, labels)
                # print("loss=", loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            eval_acc, eval_loss = self.evaluate(eval_loader)
            if if_print and (epoch+1) % print_every == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Eval Accuracy: {eval_acc}, Eval Loss: {eval_loss}")
                print("test_staff=", test_staff)
            if save_path is not None:
                torch.save({"epoch": epoch, 
                            "train_loss": running_loss/len(train_loader), 
                            "eval_loss": eval_loss,
                            "optimizer_state_dict": self.optimizer.state_dict(), 
                            "model_state_dict":self.model.state_dict()}
                           , f"{save_path}/checkpoint-{epoch}.pt")
        return eval_acc

    def evaluate(self, eval_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_loader:
                inputs = batch["input"].to(self.model_device).squeeze()
                labels = batch["label"].to(self.model_device).squeeze()
                outputs = self.model(inputs)
                loss_fct = CrossEntropyLoss()
                test_loss = loss_fct(outputs, labels)
                # print("outputs=", outputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.squeeze()
                # print("predicted=", Tokenizer.batch_decode(predicted))
                # print("labels=", Tokenizer.batch_decode(labels))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        return acc, test_loss.item()