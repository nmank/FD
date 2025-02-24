import torch
from torch import nn
from NetworkHeads import *

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module, head: str, both: bool = False, fl_type: list = [1,1]):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        self.head = head
        self.both = both
        self.fl_type = fl_type

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """

        # Extract the features of support and query images
        if self.both:
            z_support1, z_support2 = self.backbone.forward(support_images)
            z_query1, z_query2 = self.backbone.forward(query_images)
        else:
            z_support = self.backbone.forward(support_images)
            z_query = self.backbone.forward(query_images)
        
        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Infer the number shots from the labels of the support set
        n_shot = len(torch.where(support_labels==support_labels[0])[0])

        if self.both:
            if self.head == 'ProtoNet':
                scores = ProtoNetHeadMod(z_query1, z_query2, z_support1, z_support2, support_labels, n_way, n_shot)
            elif self.head == 'SubspaceNet':
                scores = SubspaceNetHeadMod(z_query1, z_query2, z_support1, z_support2, support_labels, n_way, n_shot)
            elif self.head == 'FlagNet':
                scores = FlagNetHead(z_query1, z_query2, z_support1, z_support2, support_labels, n_way, n_shot, self.fl_type)
            else:
                print('head not recognized')

        else:
            if self.head == 'ProtoNet':
                scores = ProtoNetHead(z_query, z_support, support_labels, n_way, n_shot)
            elif self.head == 'SubspaceNet':
                scores = SubspaceNetHead(z_query, z_support, support_labels, n_way, n_shot)
            else:
                print('head not recognized')

        return scores