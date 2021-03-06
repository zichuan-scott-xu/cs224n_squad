'''
Debug scripts to test the coattention model
'''

from qanet2 import QANet
from util import SQuAD, collate_fn, torch_from_json
import torch.utils.data as data
import torch.nn.functional as F
from args import get_train_args

args = get_train_args()

# Get embeddings
print('Loading embeddings...')
word_vectors = torch_from_json(args.word_emb_file)
char_vectors = torch_from_json(args.char_emb_file)

# Get model
print('Building model...')
# model = QANet(word_mat=word_vectors,
#               char_mat=char_vectors)
model = QANet(word_vectors=word_vectors,
              char_vectors=char_vectors,
              model_dim=args.qanet_hidden,
              num_model_enc_block=args.num_enc_blocks,
              num_heads=args.num_heads)

print('Building dataset...')
train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=0, # important to change it as 0, otherwise multiprocessing error
                                   collate_fn=collate_fn)

for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
    result = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
    print('res: {}'.format(result))
    # loss = F
    break
    
