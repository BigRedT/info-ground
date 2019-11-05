import torch
import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from .dataset import DetFeatDatasetConstants, DetFeatDataset
from .models.cap_encoder import CapEncoderConstants, CapEncoder


def replace_tokens(token_ids,ids_to_repl,val):
    token_ids = copy.deepcopy(token_ids)
    B = len(ids_to_repl)
    for i in range(B):
        for j in ids_to_repl[i]:
            token_ids[i][j] = val
        
    return token_ids

model_const = CapEncoderConstants()
model_const.model = 'BertForPreTraining'
cap_encoder = CapEncoder(model_const)

const = DetFeatDatasetConstants('val')
const.read_noun_tokens = True
dataset = DetFeatDataset(const)
print(len(dataset))
collate_fn = dataset.get_collate_fn()
dataloader = DataLoader(dataset,5,num_workers=3,collate_fn=collate_fn)

K = 10

for data in dataloader:
    verb_token_ids = data['verb_token_ids']
    token_ids_, tokens, token_lens = cap_encoder.tokenize_batch(data['caption'])
    mask_token_ids_ = replace_tokens(
        token_ids_,verb_token_ids,cap_encoder.mask_token_id)
    token_ids = torch.LongTensor(token_ids_)
    mask_token_ids = torch.LongTensor(mask_token_ids_)
    
    predictions = cap_encoder.model(token_ids)[0].softmax(-1)
    mask_predictions = cap_encoder.model(mask_token_ids)[0].softmax(-1)
    
    max_ids = torch.topk(predictions,K,2)[1].detach().numpy()
    mask_max_ids = torch.topk(mask_predictions,K,2)[1].detach().numpy()
    B,L,K = max_ids.shape
    for i in range(B):
        if len(verb_token_ids[i])==0:
            continue

        j = verb_token_ids[i][0]
        preds = []
        mask_preds = []
        for k in range(K):
            preds.append(
                (cap_encoder.tokenizer._convert_id_to_token(max_ids[i][j][k]),
                round(predictions[i][j][max_ids[i][j][k]].item(),3)))
            mask_preds.append(
                (cap_encoder.tokenizer._convert_id_to_token(mask_max_ids[i][j][k]),
                round(mask_predictions[i][j][mask_max_ids[i][j][k]].item(),3)))
        print('Token replaced',tokens[i][j])
        print('Pred:',preds)
        print('Mask Pred:',mask_preds)
        print('Tokens:',tokens[i])
        import pdb; pdb.set_trace()
        #token_ids = torch.tensor(tokenizer.encode(cap,add_special_tokens=True)).unsqueeze(0)