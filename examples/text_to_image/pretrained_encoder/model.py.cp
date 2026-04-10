from .tokenizer import ProtTokenizer

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import pytorch_lightning as pl

from .attentionCompression import AttentionCompression

seed = 42
#random.seed(seed)
torch.manual_seed(seed) ## otherwise, the nn.linear results are not consistent
#torch.cuda.manual_seed(seed)

class EmbeddingFromPretrained(nn.Module):
    def __init__(self,
                 #weight_file,
                 vector_size, #1024
                 embed_dir,
                 sequence_max_length, # = 10,
                 pad_token='[PAD]',
                 pad_after=True,
                 existing_words=None,
                 verbose=False):

        super(EmbeddingFromPretrained, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed_dir = embed_dir
        #self.weight_file = weight_file
        self.vector_size = vector_size # 1024
        self.sequence_max_length = sequence_max_length

        self.pad_token = pad_token
        self.pad_index = 0
        self.pad_after = pad_after

        self.existing_words = existing_words if existing_words is not None else []
        
        self.word2index = {
            self.pad_token: self.pad_index
        }
        self.index2word = {
            self.pad_index: self.pad_token
        }

        self.prot2species = {}
        self.species_embedding = [np.zeros(shape=(vector_size, ))]
        
        #self.embedding_layer = self.collect_embeddings(verbose=verbose) #load embedding of all tokens in vocabulary
        self.pretrained_embeddings = self.load_species_embedding()

    def load_species_embedding(self):
        ## total embedding files is around 20GB, so need to store/load by species
        self.prot_index = len(self.word2index) ## start from 1. 0 is reserved for [PAD]
        print(">>>> load precalculated embeddings")

        for root, dir, files in os.walk(self.embed_dir, followlinks=True):
            for file in tqdm(files):
                if file.endswith('.npy'):
                    spe_name = file.replace('.npy', '')
                    weight_file, pidx_file = os.path.join(root, file), os.path.join(root, file.replace('.npy', '.idx'))
                    self.collect_embeddings(spe_name, weight_file, pidx_file)

        
        outf = open('example_bird_protein_vocabulary.txt', 'w')
        for k,v in self.word2index.items():
            outf.write(f"{v},{k}\n")
        outf.close()
        
        return torch.nn.Embedding.from_pretrained(torch.Tensor(np.array(self.species_embedding))).to(self.device)

    def collect_embeddings(self, spe_name, weight_file, pidx_file, verbose=False, vector_size= 1024):## original use is to load entire vocabulary's word embedding (per species)        
        embedding_matrix = [np.zeros(shape=(vector_size, ))] # load embedding vectors of entire vocabulary
        print('loaing embeddings of ', spe_name)
        
        # load pre-calculated protein embedding of one species
        #npy file doesn't have protein name, only stores the embedding vector
        lines_weight = np.load(weight_file) #('./embeddings_prot_bert_bfd/Tyto_alba_Barn_owl.npy')
        
        ff= open(pidx_file, mode='r', encoding='utf-8', errors='ignore') #('./embeddings_prot_bert_bfd/Tyto_alba_Barn_owl.idx')
        pidx = ff.readlines()
        prot_names = [p.split(',')[1].split(' ')[0] for p in pidx]
        
        #index = len(self.word2index)# begin with 1
    
        for embeddings, prot in zip(lines_weight, prot_names):# for each protein embedding
            if  embeddings.shape[0] != self.vector_size: # or prot not in existing_words: ## gene not seen before 
                print('incorrect embedding, embeddings.shape[0] ' , embeddings.shape[0], vector_size)
                #break 
                continue
                
            self.word2index[prot] = self.prot_index
            self.index2word[self.prot_index] = prot
    
            embedding_matrix.append(embeddings)
            self.prot_index += 1

        self.species_embedding.extend(embedding_matrix)


    def forward(self, input_batch):
        sequence_lengths = [] ## max number of words per input
        #sequence_max_length = 10 ## 1000 proteins per input - image pair
        
        embedded_batch = torch.Tensor(size=(len(input_batch), self.sequence_max_length, self.vector_size)).to(self.device) #1024
        
        for n_sample in range(len(input_batch)):## a batch of n_sample sentences (1000 proteins per sample)
            """
            tokens = [self.word2index[token] for token in input_batch[n_sample].split(',') if token in self.word2index] #get tokens/protein in each input sentence
            tokens = tokens[:self.sequence_max_length] # tokens are a list of word index ==> ids
                
            sequence_lengths.append(len(tokens))
            if len(tokens) < self.sequence_max_length:
                pads = [self.pad_index] * (self.sequence_max_length - len(tokens))## pad with token index 0
                if self.pad_after:
                    tokens = tokens + pads
                else:
                    tokens = pads + tokens
            """
            ##input_batch is ids
            tokens = input_batch[n_sample]
            #print(">>>>", tokens)
            try:
                tokens = torch.LongTensor(tokens).to(device=self.device)
                embedded_batch[n_sample] = self.pretrained_embeddings(tokens).to(device=self.device) ## get embeddings of given tokens in input_b, save to embedded_batch
            except Exception as e :
                #print('tokens not found:', tokens)
                raise ValueError(f"invalid token: {e}")
                        
        if embedded_batch.sum() == 0:
            return None, None
        
        sequence_lengths = torch.Tensor(sequence_lengths)
        #sequence_lengths, permutation_idx = sequence_lengths.sort(descending=True)
        
        ## perhaps don't have to move to cuda 
        ## sequence_lengths = sequence_lengths.to(device=self.device)
        
        ## for calcualting the embedding of sub-proteome of brid, only need DAN network to output vectors
        ## so we don't need to provide parameter "target" for  self.pretrained_embeddings(tokens, target) 
        
        ## don't need this in embeddig birds sub-proteome
        #embedded_batch = embedded_batch[permutation_idx]

        return embedded_batch, sequence_lengths 


class DenseNetwork(pl.LightningModule): #nn.Module
    def __init__(self,
                 sizes,
                 num_classes,
                 activation_function=F.relu,
                 sigmoid_output=False):

        super(DenseNetwork, self).__init__()

        self.sizes = list(sizes)
        self.activation_function = activation_function
        self.sigmoid_output = sigmoid_output

        if self.sizes[-1] != 1 and self.sigmoid_output:## output prediction; otherwise, only build last hidden layer
            self.sizes.append(1)

        self.input_size = self.sizes[0]
        self.output_size = self.sizes[-1]
        print("input_size:", self.input_size, "output_size:", self.output_size )
        
        self.linear_1 = nn.Linear(in_features=self.sizes[0], out_features=self.sizes[1])

        if len(self.sizes) > 3:
            self.linear_2 = nn.Linear(in_features=self.sizes[1], out_features=self.sizes[2])

        if len(self.sizes) > 4:
            self.linear_3 = nn.Linear(in_features=self.sizes[2], out_features=self.sizes[3])

        if len(self.sizes) > 5:
            self.linear_4 = nn.Linear(in_features=self.sizes[3], out_features=self.sizes[4])

        self.linear_last = nn.Linear(in_features=self.sizes[-2], out_features=self.sizes[-1])
        
        self.clf_head = nn.Linear(self.sizes[-1], num_classes)

    def forward(self, x):
        x = x.to(torch.bfloat16) # for supportring bfloat16
        x = self.linear_1(x)
        x = self.activation_function(x)
        
        if len(self.sizes) > 3:
            x = self.linear_2(x)
            x = self.activation_function(x)
            
        if len(self.sizes) > 4:
            x = self.linear_3(x)
            x = self.activation_function(x)
            
        if len(self.sizes) > 5:
            x = self.linear_4(x)
            x = self.activation_function(x)
        
        x = self.linear_last(x)
        last_hidden_state = x 
        #y = torch.sigmoid(last_hidden_state)
        outs = self.clf_head(self.activation_function(x))
        
        return outs, last_hidden_state


class DAN(pl.LightningModule):
    def __init__(self,
                 embedding_layer = None,
                 weight_file_fld = None,
                 embedding_size  = 1024,
                 sizes= (1024, 768, 512),
                 activation_function = F.relu,
                 num_classes=358, ## 358 bird species
                 sequence_max_length=4000,
                 sigmoid_output = False):

        super(DAN, self).__init__()
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') ## ## not working in pl.LightningModule
        self.denselayer_size= sizes 
        self.sequence_max_length = sequence_max_length
        self.embedding_size = embedding_size
        if embedding_layer is not None:
            self.embedding_layer = embedding_layer
        elif weight_file_fld is not None:
            self.embedding_layer = EmbeddingFromPretrained(vector_size=self.embedding_size, 
                                                           embed_dir=weight_file_fld,
                                                           sequence_max_length=self.sequence_max_length )#1024
        else:
            raise ValueError('Need embedding layer or weight file')

        #self.embedding_layer = self.embedding_layer.to(self.device)
        self.fc_network = DenseNetwork(sizes=sizes,
                                    num_classes=num_classes,
                                    activation_function=activation_function,
                                    sigmoid_output=sigmoid_output).to(self.device) ## not working in pl.LightningModule

    def _prepare_emb_for_clip(self, x):
        """
        params:
            x: embeddings of input tokens, shape: (batch_size, sequence_max_length, embedding_size)
        """ 
        ## update 2025/03/15, using Attenion-as-compression to transformer output dimention to 77
        """
        n = 80
        ## dim of x : batch size, 4000 tokens, 1024 dimension
        sz = x.shape ## torch.Size([1, 4000, 1024])
        #tt = torch.Tensor(size=(sz[0], n, sz[1]//n, sz[2]))
        tt = torch.Tensor(size=(sz[0], 77, sz[1]//n, 768))
        tt = tt.to('cuda:0')
        indices = torch.tensor([i for i in range(768)])
        indices = indices.to('cuda:0')
        for i in range(sz[0]): #per batch, 1 as this example
            chk = x[i].chunk(n)
            for j in range(77):
                tt[i][j] = torch.index_select(chk[j], 1, indices) #chk[j]
        return tt.mean(dim=2)
        """
        #print('x.shape in _prepare_emb_for_clip: ', x.shape  # torch.Size([1, 4000, 1024]) )
        
        ### update 2025/05/02: not using attention compression to reduce the dimension of x. Instead, embed each pathway and key protein separately, which are separated by a comma.
        """
        compression_module = AttentionCompression(input_dim=self.embedding_size, \
                                                  output_dim=768, \
                                                  num_tokens=self.sequence_max_length, \
                                                  num_compressed_tokens = 75) # 77 (consuming 1 + 1 for start + end tokens).
        #compression_module.to('cuda:0')
        compression_module.to('cuda')
        return compression_module(x)
        """
        prot_grp = x.split(';')
        embed_prot_grp = []
        for grp in prot_grp:
            #print('grp:', grp)
            if len(grp) > 0:
                pids = grp.split(',')
                #print('grp:', grp)
                if len(pids) > 1:
                    embed_prot = pids.split(',')
                    embed_prot = torch.stack(embed_prot, dim=0)
                    embed_prot = embed_prot.mean(dim=0)
                    embed_prot_grp.append(embed_prot)
        #print('embed_prot_grp:', embed_prot_grp)

        return torch.stack(embed_prot_grp, dim=0).to('cuda:0') #torch.Size([1, N, 768]), N <=77
    

    def forward(self, tokens):
        x, _ = self.embedding_layer(tokens)
        if x == None:
            print('None tokens found in DAN:')
        
        #data_for_clip = self._prepare_emb_for_clip(x.to(torch.bfloat16))# for supportring bfloat16, data_for_clip ==> torch.Size([1, 77, 768])
        data_for_clip = self._prepare_emb_for_clip(x)

        x = x.mean(dim=1)
        self.outs, self.last_hidden_state = self.fc_network(x)

        return self.outs,  data_for_clip #self.last_hidden_state  #x[:, 0]


if __name__ == '__main__':
    pretrained_embeddings = EmbeddingFromPretrained(vector_size=1024, embed_dir = '/home/jun/work/species_genAI/example_embed')
    #pretrained_embeddings = pretrained_embeddings.to('cuda:0')

    dan = DAN(
        embedding_layer = pretrained_embeddings,
        sizes = [pretrained_embeddings.vector_size, 768, 512] ,
        num_classes = 10
    )
    
    tokens = ['tr|A0A8V5HGS3|A0A8V5HGS3_MELUD,tr|A0A8C6N6X0|A0A8C6N6X0_MELUD,tr|A0A8C6JU91|A0A8C6JU91_MELUD,tr|A0A8C6IKR4|A0A8C6IKR4_MELUD',
              'tr|A0A8C6N6K2|A0A8C6N6K2_MELUD,tr|A0A8C6J5W4|A0A8C6J5W4_MELUD,tr|A0A8C6JF57|A0A8C6JF57_MELUD,tr|A0A8C6JZZ8|A0A8C6JZZ8_MELUD',]
    #tokens = ['tr|A0A7L3F1E2|A0A7L3F1E2_9GRUI,tr|A0A7L3FWF2|A0A7L3FWF2_9GRUI,tr|A0A7L3FE22|A0A7L3FE22_9GRUI']
    voc = []
    with open('./example_bird_protein_vocabulary.txt', 'r') as f:
        for line in f:
            aline = line.strip().split(',')
            voc.append(aline[1])
    
    tokenizer = ProtTokenizer(voc ,max_length=10) ## result are same between using tokenizer and tokenizer inside forward function.
    tokens = tokenizer.encode(tokens)
    #print("?????", tokens)
    
    vec = dan(tokens['input_ids'])
    #vec = dan(tokens)

    print(vec)
