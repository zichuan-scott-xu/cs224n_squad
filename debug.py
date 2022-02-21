'''
Debug scripts to test the coattention model
'''

import models
from util import SQuAD, collate_fn, torch_from_json
import torch.utils.data as data
from args import get_train_args

args = get_train_args()

# Get embeddings
print('Loading embeddings...')
word_vectors = torch_from_json(args.word_emb_file)

# Get model
print('Building model...')
model = models.CoAttention(word_vectors=word_vectors,
                hidden_size=args.hidden_size,
                drop_prob=args.drop_prob)

print('Building dataset...')
train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=0, # important to change it as 0, otherwise multiprocessing error
                                   collate_fn=collate_fn)
for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
    print(model(cw_idxs, qw_idxs))
    break
    
