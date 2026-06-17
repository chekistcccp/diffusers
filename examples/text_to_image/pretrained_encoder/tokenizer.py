import torch
from transformers.tokenization_utils_base import BatchEncoding 

class ProtTokenizer:
    def __init__(self,vocab, max_length = 6000, model_max_length=6000):
        self.max_length = max_length
        self.model_max_length = model_max_length
        self.pad_token = "[PAD]" ## id = len(vocab) 
        self.unk_token = "[UNK]" ## id = len(vocab) +1
        self.grp_token = "[GRP]" ## token to seprate groups of protein ids, modified on 05/02/2025

        if isinstance(vocab, dict):
            self.vocab_to_index = {token: int(idx) for token, idx in vocab.items()}
        else:
            self.vocab_to_index = {token: (idx + 1) for idx, token in enumerate(vocab)} # start from 1

        self.vocab_to_index.setdefault(self.grp_token, 0)
        self.vocab_to_index.setdefault(self.pad_token, 1)
        max_existing_id = max(self.vocab_to_index.values(), default=1)
        self.vocab_to_index.setdefault(self.unk_token, max_existing_id + 1)
        self.vocab = [token for token, _ in sorted(self.vocab_to_index.items(), key=lambda item: item[1])]
        
        """
        ## modified on 3/24/2025
        """
        self.index_to_vocab = {idx: token for token, idx in self.vocab_to_index.items()}

        # Required for Hugging Face compatibility
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}
        self.special_tokens_map = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "grp_token": self.grp_token,
        }

        self.unique_added_tokens_encoder = set()
        self.unique_added_tokens_decoder = set()
        self._pad_token = self.pad_token
        self._unk_token = self.unk_token
        self._mask_token = self.pad_token
        self._cls_token = self.pad_token
        self._sep_token = self.pad_token
        self._grp_token = self.grp_token

        self._additional_special_tokens = []
        self._special_tokens_map = self.special_tokens_map
        self._special_tokens = []
        self._special_tokens_name_to_id = {}
        self._name_to_token = {}
        self._token_to_name = {}
        self._name_to_ids = {}
        self._ids_to_tokens = {}
        self._tokenizer = None  # Not needed for encoding   
        

    def __call__(self, proteome, **kwargs):
        """Make the tokenizer callable"""
        if isinstance(proteome, str):
            proteome = [proteome]
        max_length = kwargs.get("max_length")
        if max_length is None and kwargs.get("padding") == "longest":
            max_length = max((len(self.tokenize(item)) for item in proteome), default=0)
        max_length = max(max_length or self.max_length, 1)
        return self.encode(proteome, max_length=max_length)
    
    def _tokenize(self, text):
        return text.strip().split(',')

    def _convert_token_to_id(self, token):
        return self.vocab_to_index.get(token, self.vocab_to_index[self.unk_token])

    @property
    def pad_token_id(self):
        """
        ## modified on 3/24/2025
        """
        return self.vocab_to_index[self.pad_token]
    
    @property
    def unk_token_id(self):
        """
        ## modified on 3/24/2025
        """
        return self.vocab_to_index[self.unk_token]

    def tokenize(self, sentence):
        """"
        params:
            sentence: a list of protein ids seprated by comman and semicolon. semicolon is used to seprate different groups of protein ids
        return: a list of token ids, and the location of group separator(;)
        """

        sentence = sentence.strip()
        if not sentence:
            return []
        return sentence.split(',')
    
    def get_vocab(self):
        return self.vocab_to_index

    def pad_and_truncate(self, token_ids, max_length=None):
        max_length = max_length or self.max_length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            pad_length = max_length - len(token_ids)
            token_ids += [self.pad_token] * pad_length
        #print(token_ids)
        return token_ids

    def encode(self, proteome, max_length=None):
        max_length = max_length or self.max_length
        ids, token_type_ids, att_mask, grp_sep = [],[],[],[]
        for p in proteome:
            tk = self.tokenize(p)# tk contains self.grp_token
            mask = [1]* min(max_length,len(tk))  # changed on 05/02/2025
            
            ##TODO: maybe should not separate them here. ==> Updated on 05/024/2025, implemeted in model.py:291-306            
            #mask = [1 for i in range(1 + len(grp_sep) )] # changed on 05/02/2025, one mask per group
            tk = self.pad_and_truncate(tk, max_length=max_length) # changed on 05/02/2025, no need to pad and truncate here since we are using the group separator. We will pad and truncate later
            #if len(mask) < (len(tk)):
            #    mask.extend( [0]*(len(tk) -len(mask)) )

            if len(mask) < max_length:
                mask.extend( [0]*(max_length -len(mask)) )

            """
            modified on 3/25/2025
            ids.append([ int(self.vocab_to_index[t]) if t in self.vocab_to_index else self.vocab_to_index[self.unk_token]  for t in tk ])
            #if ids[0][0] == self.unk_token:
            #    print(p)
            token_type_ids.append([0]*len(ids))
            """

            # Convert tokens to IDs. modified on 3/25/2025. 05/03: including separator token
            sequence_ids = [
                int(self.vocab_to_index[t]) if t in self.vocab_to_index 
                else self.vocab_to_index[self.unk_token] 
                for t in tk
            ]
            ids.append(sequence_ids)
            # Create token_type_ids with same length as the token sequence
            token_type_ids.append([0] * len(tk))

            att_mask.append(mask)
            
        ## modified on 3/24/2025
        ##return {"input_ids":ids, 
        #        "token_type_ids": [0]*len(ids),
        #        "attention_mask": att_mask}

        ## modified on 3/24/2025
        # Convert to BatchEncoding-like object
        #print('<><><><> tokenizer line 107: ', ids, len(ids[0]))
        #return type('BatchEncoding', (), {
        #    'input_ids': torch.tensor(ids),
        #    'token_type_ids': torch.tensor(token_type_ids),
        #    'attention_mask': torch.tensor(att_mask)
        #    })()
        batch_outputs = BatchEncoding(
            data={
                "input_ids": torch.tensor(ids),
                "token_type_ids": torch.tensor(token_type_ids),
                "attention_mask": torch.tensor(att_mask), # not including the group separator
            },
            tensor_type="pt",  # Specify PyTorch tensors
        )
        return batch_outputs

    def batch_decode(self, sequences, skip_special_tokens=True):
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()

        decoded = []
        special_tokens = {self.pad_token} if skip_special_tokens else set()
        for sequence in sequences:
            tokens = []
            for idx in sequence:
                token = self.index_to_vocab.get(int(idx), self.unk_token)
                if token in special_tokens:
                    continue
                tokens.append(token)
            decoded.append(",".join(tokens))
        return decoded

if __name__ == "__main__":
    voc = {}
    with open('example_bird_protein_vocabulary.txt', 'r') as f:
        for line in f:
            aline = line.strip().split(',', 1)
            voc[aline[1]] = int(aline[0])

    tokenizer = ProtTokenizer(voc ,max_length=10)
    #p = "tr|A0A7L3F1E2|A0A7L3F1E2_9GRUI,tr|A0A7L3FWF2|A0A7L3FWF2_9GRUI,tr|A0A7L3FE22|A0A7L3FE22_9GRUI"
    p = ['tr|A0A8C6J125|A0A8C6J125_MELUD,tr|A0A8V5GTX0|A0A8V5GTX0_MELUD,[GRP],tr|A0A8C6K1U5|A0A8C6K1U5_MELUD,tr|A0A8V5FQ45|A0A8V5FQ45_MELUD']
    encoded_text = tokenizer.encode(p)
    print(encoded_text)
