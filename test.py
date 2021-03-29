import gensim
import torch
import torch.nn.functional as F


att = torch.FloatTensor([[[1, 2, 3],
                          [1, 2, 3],
                          [1, 2, 3]],
                         [[1, 2, 3],
                          [1, 2, 3],
                          [1, 2, 3]]]
                        )

att_score = F.softmax(att, dim=2)
print(att_score)


