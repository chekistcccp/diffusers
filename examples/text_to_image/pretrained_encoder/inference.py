from .model import DAN, EmbeddingFromPretrained
from .tokenizer import ProtTokenizer

import json
import torch
import pytorch_lightning as pl
import yaml
import argparse
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace


def load_vocab_mapping(voc_dir):
    vocab_to_index = OrderedDict()
    with open(voc_dir, 'r') as f:
        for line in f:
            idx, token = line.strip().split(',', 1)
            vocab_to_index[token] = int(idx)
    return vocab_to_index

class speciesModel(pl.LightningModule):
    def __init__(self, max_tokens,
                 embeddings,
                 species_classes,
                 voc_dir,
                 checkpoint_path=None,
                 device=None,
                 ):
        super(speciesModel, self).__init__()

        max_tokens = int(max_tokens)
        self.runtime_device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Keep the custom species encoder in float32 to avoid dtype mismatches
        # across the embedding layer, compression module, and dense classifier.
        self.runtime_dtype = torch.float32
        
        """
        ## modified on 3/24/2025
        """
        # Store config parameters as instance attributes
        self.config = SimpleNamespace(
            max_tokens=int(max_tokens),
            embeddings=embeddings,
            species_classes=species_classes,
            vocabulary=voc_dir,
            use_attention_mask=True,
        )

        self.species_id = '' 
        with open(species_classes, 'r' ) as f:
            self.species_id = json.load(f)

        vocab_to_index = load_vocab_mapping(voc_dir)

        try:
            embedding_layer = EmbeddingFromPretrained(
                vector_size=1024,
                embed_dir=embeddings,
                sequence_max_length=max_tokens,
                device=self.runtime_device,
            ).to(self.runtime_device)
        except TypeError:
            # Backward-compatible fallback for environments still using the older
            # EmbeddingFromPretrained signature without an explicit device argument.
            embedding_layer = EmbeddingFromPretrained(
                vector_size=1024,
                embed_dir=embeddings,
                sequence_max_length=max_tokens,
            ).to(self.runtime_device)
        
        self.tokenizer = ProtTokenizer(vocab_to_index, max_length=max_tokens, model_max_length=max_tokens) ## result are same between using tokenizer and tokenizer inside forward function.

        self.model = DAN(sizes=[768, 256, 128], # sync up with ~/work/species_genAI/finetune/pretrain_species_encoder_susbsystems/trainer.py:62
                            embedding_layer=embedding_layer,
                            sequence_max_length=max_tokens,
                            num_classes = len(self.species_id) #358
        ).to(self.runtime_device)
        
        #self.model = DAN.load_from_checkpoint('./logs/ckpts/epoch=5-step=13125.ckpt', 
        #                                      embedding_layer=embedding_layer,
        #)

        if checkpoint_path is None:
            raise ValueError("species encoder checkpoint path is required")
        self.load_model(checkpoint_path)
        print('loaded model')

    def load_model(self, ckpt_path):
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"species encoder checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        remove_prefix = 'encoder.'
        #checkpoint = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in checkpoint["state_dict"].items()}
        state_dict = OrderedDict([ 
            (k[len(remove_prefix):], v) if k.startswith(remove_prefix) else (k , v) for k, v in checkpoint["state_dict"].items()
                                    ]
                                )

        model_state_dict = self.model.state_dict()
        filtered_state_dict = OrderedDict()
        skipped_keys = []

        for key, value in state_dict.items():
            if key not in model_state_dict:
                skipped_keys.append((key, "missing_in_current_model", tuple(value.shape), None))
                continue

            current_shape = tuple(model_state_dict[key].shape)
            checkpoint_shape = tuple(value.shape)
            if current_shape != checkpoint_shape:
                skipped_keys.append((key, "shape_mismatch", checkpoint_shape, current_shape))
                continue

            filtered_state_dict[key] = value

        incompatible = self.model.load_state_dict(filtered_state_dict, strict=False)

        if skipped_keys:
            print("Skipped incompatible checkpoint parameters during non-strict load:")
            for key, reason, checkpoint_shape, current_shape in skipped_keys:
                print(
                    f"  - {key}: {reason}, checkpoint_shape={checkpoint_shape}, current_shape={current_shape}"
                )

        if incompatible.missing_keys:
            print(f"Missing keys after non-strict load: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys after non-strict load: {incompatible.unexpected_keys}")
        #epoc = checkpoint['epoch']
        #loss = checkpoint['loss']
        #for param in self.model.parameters():
        #    param.data = param.data.to(torch.bfloat16)

        self.model.to(self.runtime_device, dtype=self.runtime_dtype)# for supportring bfloat16
        self.model.eval()

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        return next(self.model.parameters()).device
        
    def forward(self, x, attention_mask=None, **kwargs ):
        """
        params:
            x: tokens['input_ids']
            
        """
        x = x.to(self.runtime_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.runtime_device)
        logits, last_hidden_states = self.model(x, attention_mask=attention_mask)
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
        return self.forward(tokens['input_ids'], attention_mask=tokens['attention_mask'])


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

    checkpoint_path = config.get('checkpoint') or config.get('encoder_checkpoint')
    model = speciesModel(
        config['max_tokens'],
        config['embeddings'],
        config['species_classes'],
        config['vocabulary'],
        checkpoint_path=checkpoint_path,
    )
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
