from overrides import overrides

import torch
import torch.nn

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

@Seq2VecEncoder.register("max_pooler_seq2vec")
class max_pooler_seq2vec(Seq2VecEncoder):

    def __init__(self, requires_grad: bool = True, embedding_dim: int = 768 , dropout: float = 0.0) -> None:
        super().__init__()

        self._dropout = torch.nn.Dropout(p=dropout)
        self._embedding_dim = embedding_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor = None):
        #pooled = self.pooler(tokens)
        """
        tokens are batch x num_chunks x embedding_dim
        returns a maximum embedding value over chunks. 
        """

        assert tokens.shape[2] == self._embedding_dim
        pooled = tokens.max(dim=1)[0] # the [0] takes only the values and drops indices, this is automatically squeezed.
        pooled = self._dropout(pooled)
        return pooled