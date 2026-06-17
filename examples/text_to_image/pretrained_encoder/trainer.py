import json
import pickle 
import math

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from argparse import ArgumentParser
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,auc
)

from model import DAN, EmbeddingFromPretrained
from tokenizer import ProtTokenizer

def learning_rate_schedule(warmup_steps, total_steps):
    def learning_rate_fn(step):
        if step< warmup_steps:
            return float(step)/float(max(1, warmup_steps))
        else:
            progress = float(step-warmup_steps)/float(max(1, total_steps-warmup_steps))
            return 0.5*(1.0 + math.cos(math.pi * progress))
        
    return learning_rate_fn

class birdClassifier(pl.LightningModule):
    def __init__(self, 
                 learning_rate=3e-3, 
                 steps=40000, 
                 warmup_steps=4000,
                 **kwargs,):
        
        super(birdClassifier,self).__init__()

        self.save_hyperparameters()
        max_tokens = int(kwargs['max_tokens'])

        self.species_id = '' 
        with open(kwargs['species_classes'], 'r' ) as f:
            self.species_id = json.load(f)
        
        embedding_layer = EmbeddingFromPretrained(vector_size=1024,
                                                embed_dir = kwargs["embeddings"],
                                                species_id=self.species_id,
                                                voc_file=kwargs['vocabulary'],
                                                sequence_max_length= max_tokens )
        #embedding_layer =  embedding_layer.to('cpu')  #to('cuda:0')

        voc = []
        #with open('./example_bird_protein_vocabulary.txt', 'r') as f:
        with open(kwargs['vocabulary'], 'r') as f:
            for line in f:
                aline = line.strip().split(',')
                voc.append(aline[1])
        
        self.tokenizer = ProtTokenizer(voc ,max_length = max_tokens) ## result are same between using tokenizer and tokenizer inside forward function.
        self.encoder = DAN(
                    #sizes=[embedding_layer.vector_size, 768, 512],
                    # or maybe try sizes=[ 768, 1024, 512]
                    embedding_layer = embedding_layer,
                    species_id=self.species_id,
                    sizes=[768, 256, 128], # comply with the code "self.outs, self.last_hidden_state = self.fc_network(data_for_clip)" in model.py
                    sequence_max_length = max_tokens,
                    num_classes = len(self.species_id) #358
        )## output is last hidden layer
        

        self.encoder.to('cuda:0')
        self.clf_criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.validation_step_outputs = []
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": lr_scheduler.LambdaLR(
                optimizer, learning_rate_schedule(self.warmup_steps, self.steps)
            ),
            "interval": "step",
        }
        
        return [optimizer], [scheduler]
    
    
    """
    According to PyTorch documentation:
    The input of nn.CrossEntropyLoss() is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1, in general).
    Therefore it is recommended to set the raw logit instead of probabilities.
    """
    def training_step(self, batch):
        x, labels = batch
        #print(x[0])
        tokens = self.tokenizer.encode(x)
        clf_logits, last_hidden_state = self.encoder(tokens['input_ids'])
        #print(clf_logits, labels)
        score = torch.softmax(clf_logits, dim=1)
        labels = torch.tensor([int(x) for x in labels]).to('cuda:0')
        clf_loss = self.clf_criterion(clf_logits, labels)
        #clf_loss = self.clf_criterion(score, labels)
        return {"loss": clf_loss}
    
    def validation_step(self, batch):
        x,labels = batch
        tokens = self.tokenizer.encode(x)
        clf_logits, last_hidden_state  = self.encoder(tokens['input_ids'])
        #print(clf_logits, labels)
        score = torch.softmax(clf_logits, dim=1)
        labels = torch.tensor([int(x) for x in labels]).to('cuda:0')
        clf_loss = self.clf_criterion(clf_logits, labels)
        #clf_loss = self.clf_criterion(score, labels)

        #score = torch.softmax(clf_logits, dim=1)[:,1]
        _, preds = torch.max(clf_logits,1)

        correct = preds == labels
        outs = {
            "val_loss": clf_loss,
            "correct": correct,
            "preds": preds,
            "targets": labels,
            "score": score
        }
        self.validation_step_outputs.append(outs)

        return outs
    
    def on_validation_epoch_end(self):
        """
        Compute the validation and ROC metrics
        """
        outs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        logs = {"val_loss": avg_loss}
        target_names = list(self.species_id.keys())  #['rogon_melanurus', 'Zapornia_atra', 'Calypte_anna']
        print('<><><><> target names:', target_names)
        targets  = torch.cat([x["targets"] for x in outs]).cpu().numpy()
        preds    = torch.cat([x["preds"]   for x in outs]).cpu().numpy()
        correct  = torch.cat([x["correct"]  for x in outs]).cpu().numpy()

        ##multi-class classification reports
        ## In a multi-class classification setup with highly imbalanced classes, micro-averaging is preferable over macro-averaging. 
        ## In such cases, one can alternatively use a weighted macro-averaging, not demoed here.

        print('\n<><><><> targets:',targets )
        print('\n<><><><> preds:',preds )
        
        # Provide the full range of class labels so target_names matches the expected number of classes,
        # even if some classes are not present in this validation epoch.
        print(classification_report(
            targets,
            preds,
            labels=list(range(len(target_names))),
            target_names=target_names,
            zero_division=0
        ))

        # clear stored outputs so they don't accumulate across epochs
        self.validation_step_outputs = []

        """
        # Compute micro-average ROC curve and ROC area
        micro_roc_auc_ovr = roc_auc_score(preds,targets, multi_class="ovr", average="micro")
        print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")

        fpr, tpr, roc_auc = dict(), dict(), dict()
        fpr["micro"], tpr["micro"], thresholds = roc_curve(preds.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        metrics_keys = [
            "fpr", "tpr", "roc", "roc_thresholds"
        ]
        metrics_val = [fpr["micro"], tpr["micro"],roc_auc["micro"],  thresholds]
        metrics_dict = dict(zip(metrics_keys, metrics_val))

        with  open("metrics_log.pickle", "wb") as handle:
            pickle.dump(metrics_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
        """

        logs["val_acc"] = correct.sum() / correct.shape[0]
        return {"val_loss": avg_loss, "log": logs}

    def test_step(self, batch):
        return self.validation_step(batch)

    #Support for `test_epoch_end` has been removed in v2.0.0. `birdClassifier` implements this method. You can use the `on_test_epoch_end` hook instead. 
    #To access outputs, save them in-memory as instance attributes. You can find migration examples in https://github.com/Lightning-AI/lightning/pull/16520
    def on_test_epoch_end(self):
        result = self.on_validation_epoch_end()

        result["log"]["test_loss"] = result["log"].pop("val_loss")
        result["test_loss"] = result.pop("val_loss")
        if self.hparams.classify:
            result["log"]["test_acc"] = result["log"].pop("val_acc")

        return result
