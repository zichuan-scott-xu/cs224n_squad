import numpy as np

data = np.load('data/train.npz')
np.savez('data/smaller_train.npz', context_idxs=data['context_idxs'][:3], 
                                    context_char_idxs=data['context_char_idxs'][:3], 
                                    ques_idxs=data['ques_idxs'][:3], 
                                    ques_char_idxs=data['ques_char_idxs'][:3], 
                                    y1s=data['y1s'][:3], 
                                    y2s=data['y2s'][:3], 
                                    ids=data['ids'][:3])