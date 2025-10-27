from .model import DAN, EmbeddingFromPretrained
from .tokenizer import ProtTokenizer

import json
import torch
import pytorch_lightning as pl
import yaml
import argparse
from collections import OrderedDict

class speciesModel(pl.LightningModule):
    def __init__(self, max_tokens,
                 embeddings,
                 species_classes,
                 voc_dir,
                 ):
        super(speciesModel, self).__init__()

        max_tokens = int(max_tokens)
        #self.learning_rate = kwargs['learning_rate']
        
        """
        ## modified on 3/24/2025
        """
        # Store config parameters as instance attributes
        self.config = {
            'max_tokens': int(max_tokens),
            'embeddings': embeddings,
            'species_classes': species_classes,
            'vocabulary': voc_dir
        }

        self.species_id = '' 
        with open(species_classes, 'r' ) as f:
            self.species_id = json.load(f)

        voc = []
        with open(voc_dir , 'r') as f: # './example_bird_protein_vocabulary.txt'
            for line in f:
                aline = line.strip().split(',')
                voc.append(aline[1])

        embedding_layer = EmbeddingFromPretrained(
            vector_size=1024, 
            embed_dir = embeddings,
            sequence_max_length= max_tokens 
            ).to('cuda')
        
        self.tokenizer = ProtTokenizer(voc ,max_length= max_tokens) ## result are same between using tokenizer and tokenizer inside forward function.

        self.model = DAN(sizes=[768, 256, 128], # sync up with ~/work/species_genAI/finetune/pretrain_species_encoder_susbsystems/trainer.py:62
                            embedding_layer=embedding_layer,
                            sequence_max_length=max_tokens,
                            num_classes = len(self.species_id) #358
        )
        
        #self.model = DAN.load_from_checkpoint('./logs/ckpts/epoch=5-step=13125.ckpt', 
        #                                      embedding_layer=embedding_layer,
        #)

        ## run species image training
        #load retrained encoder model using grouped proteins
        #self.load_model('/home/jun/work/species_genAI/finetune/pretrain_species_encoder/logs/ckpts/epoch=5-step=13125.ckpt')
        self.load_model('/home/jun/work/species_genAI/finetune/pretrain_species_encoder_susbsystems/logs/ckpts/last.ckpt')
        print('loaded model')

    def load_model(self, ckpt_path):
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        checkpoint = torch.load(ckpt_path)
        remove_prefix = 'encoder.'
        #checkpoint = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in checkpoint["state_dict"].items()}
        state_dict = OrderedDict([ 
            (k[len(remove_prefix):], v) if k.startswith(remove_prefix) else (k , v) for k, v in checkpoint["state_dict"].items()
                                    ]
                                )
        # 使用非严格模式加载模型参数，以处理新增的局部稀疏注意力层
        self.model.load_state_dict(state_dict, strict=False)
        #epoc = checkpoint['epoch']
        #loss = checkpoint['loss']
        #for param in self.model.parameters():
        #    param.data = param.data.to(torch.bfloat16)

        self.model.to('cuda', dtype=torch.bfloat16)# for supportring bfloat16
        self.model.eval()
        
    def forward(self, x, attention_mask=None, **kwargs ):
        """
        params:
            x: tokens['input_ids']
            
        """
        if x.device.type == 'cuda': ## for difuserion model only
            x = x.detach().cpu()
        logits, last_hidden_states = self.model(x)
        score = torch.softmax(logits, dim=1)
        probs, preds = torch.max(score,1)
        
        #print(score, preds)
        #results.append([score, preds])
        #h = torch.cat([last_hidden_states]).detach().cpu().numpy()
        
        """
        #p = torch.cat([probs]).detach().cpu().numpy() ## tmpory comment off for supportring bfloat16
        y = torch.cat([preds]).cpu().numpy()
        return y, last_hidden_states
        """

        ## modified on 3/24/2025
        # Return the hidden states as tensor instead of numpy array
        return  [last_hidden_states]  # Remove numpy conversion
    
    def predict(self, test_data): ## input is raw tokens
        #results = []
        #for line in test_data:
            #row = x.split('\t')
        #x,label = test_data
        tokens = self.tokenizer.encode(test_data) # tokens are ids
        return self.forward(tokens['input_ids'])


def test(args):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    #os.environ["CUDA_VISIBLE_DEVICES"]="0";

    with open(args.config, 'rb') as f:
        config = yaml.safe_load(f)

    #test_ds = speciesData(args.dataset, test=True)
    #test_dl = DataLoader(test_ds, shuffle=True, batch_size = config['batch_size'], num_workers = config['num_workers'], drop_last = True)

    testdata = []
    with open(args.dataset, 'r') as ff:
        for line in ff:
            #tmp = line.strip().split('\t')
            testdata.append(line.strip().split('\t')[1])

    model = speciesModel(config['max_tokens'], config['embeddings'], config['species_classes'], config['vocabulary'])
    #model.load_model(args.checkpoint)
    last_hidden, y = model.predict(testdata)
    #print(last_hidden, y)
    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="")
    
    subparsers = parser.add_subparsers()

    parser_test = subparsers.add_parser("test")
    parser_test.add_argument("--checkpoint", type=str)
    parser_test.add_argument("--config", type=str)
    parser_test.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)
