import torch


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies


def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    
    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))


def SubspaceNetHead(query, support, support_labels, n_way, n_shot, normalize=True):
    """
       Constructs the subspace representation of each class(=mean of support vectors of each class) and
       returns the classification score (=L2 distance to each class prototype) on the query set.

        Our algorithm using subspaces here
        https://github.com/chrysts/dsn_fewshot/blob/master/Resnet12/models/classification_heads.py

       Parameters:
         query:  a (tasks_per_batch, n_query, d) Tensor.
         support:  a (tasks_per_batch, n_support, d) Tensor.
         support_labels: a (tasks_per_batch, n_support) Tensor.
         n_way: a scalar. Represents the number of classes in a few-shot classification task.
         n_shot: a scalar. Represents the number of support examples given per class.
         normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
       Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    query = query.unsqueeze(0)
    support = support.unsqueeze(0)
    support_labels = support_labels.unsqueeze(0)

    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    support_reshape = support.view(tasks_per_batch * n_support, -1)

    support_labels_reshaped = support_labels.contiguous().view(-1)
    class_representatives = []
    for nn in range(n_way):
        idxss = (support_labels_reshaped == nn).nonzero()
        all_support_perclass = support_reshape[idxss, :]
        class_representatives.append(all_support_perclass.view(tasks_per_batch, n_shot, -1))

    class_representatives = torch.stack(class_representatives)
    class_representatives = class_representatives.transpose(0, 1) #tasks_per_batch, n_way, n_support, -1
    class_representatives = class_representatives.transpose(2, 3).contiguous().view(tasks_per_batch*n_way, -1, n_shot)

    dist = []
    for cc in range(tasks_per_batch*n_way):
        batch_idx = cc//n_way
        qq = query[batch_idx]
        uu, _, _ = torch.svd(class_representatives[cc].double())
        uu = uu.float()
        subspace = uu[:, :n_shot-1].transpose(0, 1)
        projection = subspace.transpose(0, 1).mm(subspace.mm(qq.transpose(0, 1))).transpose(0, 1)
        dist_perclass = torch.sum((qq - projection)**2, dim=-1)
        dist.append(dist_perclass)

    dist = torch.stack(dist).view(tasks_per_batch, n_way, -1).transpose(1, 2)
    logits = -dist

    if normalize:
        logits = logits / d
    
    return logits


def ProtoNetHead(query, support, support_labels, n_way, n_shot, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and 
    returns the classification score (=L2 distance to each class prototype) on the query set.
    
    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    query = query.unsqueeze(0)
    support = support.unsqueeze(0)
    support_labels = support_labels.unsqueeze(0)

    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    
    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    
    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    #************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1,2)

    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    # Distance Matrix Vectorization Trick
    AB = computeGramMatrix(query, prototypes)
    AA = (query * query).sum(dim=2, keepdim=True)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
    logits = -logits

    
    if normalize:
        logits = logits / d

    return logits


def SubspaceNetHeadMod(query1, query2, support1, support2, support_labels, n_way, n_shot, normalize=True):
    """
       Constructs the subspace representation of each class(=mean of support vectors of each class) and
       returns the classification score (=L2 distance to each class prototype) on the query set.

        Our algorithm using subspaces here
        https://github.com/chrysts/dsn_fewshot/blob/master/Resnet12/models/classification_heads.py
   
       Parameters:
         query:  a (tasks_per_batch, n_query, d) Tensor.
         support:  a (tasks_per_batch, n_support, d) Tensor.
         support_labels: a (tasks_per_batch, n_support) Tensor.
         n_way: a scalar. Represents the number of classes in a few-shot classification task.
         n_shot: a scalar. Represents the number of support examples given per class.
         normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
       Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    query1 = query1.unsqueeze(0)
    query2 = query2.unsqueeze(0)
    support1 = support1.unsqueeze(0)
    support2 = support2.unsqueeze(0)
    support_labels = support_labels.unsqueeze(0)

    support = torch.cat([support1, support2], axis = 2)
    query = torch.cat([query1, query2], axis = 2)

    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    support_reshape = support.view(tasks_per_batch * n_support, -1)

    support_labels_reshaped = support_labels.contiguous().view(-1)
    class_representatives = []
    for nn in range(n_way):
        idxss = (support_labels_reshaped == nn).nonzero()
        all_support_perclass = support_reshape[idxss, :]
        class_representatives.append(all_support_perclass.view(tasks_per_batch, n_shot, -1))

    class_representatives = torch.stack(class_representatives)
    class_representatives = class_representatives.transpose(0, 1) #tasks_per_batch, n_way, n_support, -1
    class_representatives = class_representatives.transpose(2, 3).contiguous().view(tasks_per_batch*n_way, -1, n_shot)

    dist = []
    for cc in range(tasks_per_batch*n_way):
        batch_idx = cc//n_way
        qq = query[batch_idx]
        uu, _, _ = torch.svd(class_representatives[cc].double())
        uu = uu.float()
        subspace = uu[:, :n_shot-1].transpose(0, 1)
        projection = subspace.transpose(0, 1).mm(subspace.mm(qq.transpose(0, 1))).transpose(0, 1)
        dist_perclass = torch.sum((qq - projection)**2, dim=-1)
        dist.append(dist_perclass)

    dist = torch.stack(dist).view(tasks_per_batch, n_way, -1).transpose(1, 2)
    logits = -dist

    if normalize:
        logits = logits / d
    
    return logits


def ProtoNetHeadMod(query1, query2, support1, support2, support_labels, n_way, n_shot, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and 
    returns the classification score (=L2 distance to each class prototype) on the query set.
    
    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    query1 = query1.unsqueeze(0)
    query2 = query2.unsqueeze(0)
    support1 = support1.unsqueeze(0)
    support2 = support2.unsqueeze(0)
    support_labels = support_labels.unsqueeze(0)

    tasks_per_batch = query1.size(0)
    n_support = support1.size(1)
    n_query = query1.size(1)
    d = query1.size(2)

    
    assert(query1.dim() == 3)
    assert(support1.dim() == 3)
    assert(query1.size(0) == support1.size(0) and query1.size(2) == support1.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    
    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    #************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1,2)

    support = torch.cat([support1, support2], axis = 2)
    query = torch.cat([query1, query2], axis = 2)

    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    # Distance Matrix Vectorization Trick
    AB = computeGramMatrix(query, prototypes)
    AA = (query * query).sum(dim=2, keepdim=True)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
    logits = -logits

    
    if normalize:
        logits = logits / d

    return logits
    
#swapped
def FlagNetHead(query2, query1, support2, support1, support_labels, n_way, n_shot, fl_type = [1,1], normalize=True):
    """
       Constructs the flag representation of each class
       (=1d subspace for 2nd to last hidden layer output and 2nd subspace for last hidden layer output)
       and returns the classification score 
       (=L2 distance between projection onto flag and query pt) 
       on the query set.


       Parameters:
         query1:  a (tasks_per_batch, n_query, d) Tensor.
         support1:  a (tasks_per_batch, n_support, d) Tensor.
         query2:  a (tasks_per_batch, n_query, d) Tensor.
         support2:  a (tasks_per_batch, n_support, d) Tensor.
         support_labels: a (tasks_per_batch, n_support) Tensor.
         n_way: a scalar. Represents the number of classes in a few-shot classification task.
         n_shot: a scalar. Represents the number of support examples given per class.
         normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
       Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    query1 = query1.unsqueeze(0)
    support1 = support1.unsqueeze(0)
    query2 = query2.unsqueeze(0)
    support2 = support2.unsqueeze(0)
    support_labels = support_labels.unsqueeze(0)

    tasks_per_batch = query1.size(0)
    n_support = support1.size(1)
    n_query = query1.size(1)
    d = query1.size(2)

    assert(query1.dim() == 3)
    assert(support1.dim() == 3)
    assert(query1.size(0) == support1.size(0) and query1.size(2) == support1.size(2))
    assert(query1.size(0) == query2.size(0) and query1.size(1) == query2.size(1) and query1.size(2) == query2.size(2))
    assert(support1.size(0) == support2.size(0) and support1.size(1) == support2.size(1) and support1.size(2) == support2.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot


    support1_reshape = support1.view(tasks_per_batch * n_support, -1)
    support2_reshape = support2.view(tasks_per_batch * n_support, -1)

    support_labels_reshaped = support_labels.contiguous().view(-1)
    class_representatives1 = []
    class_representatives2 = []
    for nn in range(n_way):
        idxss = (support_labels_reshaped == nn).nonzero()
        all_support1_perclass = support1_reshape[idxss, :]
        all_support2_perclass = support2_reshape[idxss, :]
        class_representatives1.append(all_support1_perclass.view(tasks_per_batch, n_shot, -1)) 
        class_representatives2.append(all_support2_perclass.view(tasks_per_batch, n_shot, -1)) 

    class_representatives1 = torch.stack(class_representatives1)
    class_representatives1 = class_representatives1.transpose(0, 1) #tasks_per_batch, n_way, n_support, -1
    class_representatives1= class_representatives1.transpose(2, 3).contiguous().view(tasks_per_batch*n_way, -1, n_shot)

    class_representatives2 = torch.stack(class_representatives2)
    class_representatives2 = class_representatives2.transpose(0, 1) #tasks_per_batch, n_way, n_support, -1
    class_representatives2 = class_representatives2.transpose(2, 3).contiguous().view(tasks_per_batch*n_way, -1, n_shot)

    dist = []
    for cc in range(tasks_per_batch*n_way):
        batch_idx = cc//n_way

        qq1 = query1[batch_idx]
        qq2 = query2[batch_idx]

        # get subspace for last fcc output
        # uu2, ss2, _ = torch.svd(class_representatives2[cc].double())
        # m2 = torch.sum(~torch.isclose(ss2.float(),torch.tensor(0.0)))
        # uu2 = uu2.float()[:, :m2]
        uu2, _, _ = torch.svd(class_representatives2[cc].double())
        # uu2 = uu2.float()[:, [0]]
        uu2 = uu2.float()[:, :fl_type[0]]
        subspace2 = uu2.transpose(0, 1)
        
        

        # get subspace for second to last fcc output
        proj_cr = (torch.eye(d).cuda() - uu2.mm(uu2.transpose(0, 1))).mm(class_representatives1[cc])
        # uu1, ss1, _ = torch.svd(proj_cr.double())
        # m1 = torch.sum(~torch.isclose(ss1.float(),torch.tensor(0.0)))
        # uu1 = uu1.float()[:, :m1]
        uu1, _, _ = torch.svd(proj_cr.double())
        # uu1 = uu1.float()[:, [0]]
        uu1 = uu1.float()[:, :fl_type[1]]
        subspace1 = uu1.transpose(0, 1)


        projection1 = subspace1.transpose(0, 1).mm(subspace1.mm(qq1.transpose(0, 1))).transpose(0, 1)
        projection2 = subspace2.transpose(0, 1).mm(subspace2.mm(qq2.transpose(0, 1))).transpose(0, 1)
        dist_perclass = torch.sum((qq1 - projection1)**2, dim=-1) + torch.sum((qq2 - projection2)**2, dim=-1)
        dist.append(dist_perclass)

    dist = torch.stack(dist).view(tasks_per_batch, n_way, -1).transpose(1, 2)
    logits = -dist

    if normalize:
        logits = logits / d
    
    return logits

# def FlagNetHead(query1, query2, support1, support2, support_labels, n_way, n_shot, fl_type = [1,1], normalize=True):
#     """
#        Constructs the flag representation of each class
#        (=1d subspace for 2nd to last hidden layer output and 2nd subspace for last hidden layer output)
#        and returns the classification score 
#        (=L2 distance between projection onto flag and query pt) 
#        on the query set.


#        Parameters:
#          query1:  a (tasks_per_batch, n_query, d) Tensor.
#          support1:  a (tasks_per_batch, n_support, d) Tensor.
#          query2:  a (tasks_per_batch, n_query, d) Tensor.
#          support2:  a (tasks_per_batch, n_support, d) Tensor.
#          support_labels: a (tasks_per_batch, n_support) Tensor.
#          n_way: a scalar. Represents the number of classes in a few-shot classification task.
#          n_shot: a scalar. Represents the number of support examples given per class.
#          normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
#        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
#     """

#     query1 = query1.unsqueeze(0)
#     support1 = support1.unsqueeze(0)
#     query2 = query2.unsqueeze(0)
#     support2 = support2.unsqueeze(0)
#     support_labels = support_labels.unsqueeze(0)

#     tasks_per_batch = query1.size(0)
#     n_support = support1.size(1)
#     n_query = query1.size(1)
#     d = query1.size(2)

#     assert(query1.dim() == 3)
#     assert(support1.dim() == 3)
#     assert(query1.size(0) == support1.size(0) and query1.size(2) == support1.size(2))
#     assert(query1.size(0) == query2.size(0) and query1.size(1) == query2.size(1) and query1.size(2) == query2.size(2))
#     assert(support1.size(0) == support2.size(0) and support1.size(1) == support2.size(1) and support1.size(2) == support2.size(2))
#     assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot


#     support1_reshape = support1.view(tasks_per_batch * n_support, -1)
#     support2_reshape = support2.view(tasks_per_batch * n_support, -1)

#     support_labels_reshaped = support_labels.contiguous().view(-1)
#     class_representatives1 = []
#     class_representatives2 = []
#     for nn in range(n_way):
#         idxss = (support_labels_reshaped == nn).nonzero()
#         all_support1_perclass = support1_reshape[idxss, :]
#         all_support2_perclass = support2_reshape[idxss, :]
#         class_representatives1.append(all_support1_perclass.view(tasks_per_batch, n_shot, -1)) 
#         class_representatives2.append(all_support2_perclass.view(tasks_per_batch, n_shot, -1)) 

#     class_representatives1 = torch.stack(class_representatives1)
#     class_representatives1 = class_representatives1.transpose(0, 1) #tasks_per_batch, n_way, n_support, -1
#     class_representatives1= class_representatives1.transpose(2, 3).contiguous().view(tasks_per_batch*n_way, -1, n_shot)

#     class_representatives2 = torch.stack(class_representatives2)
#     class_representatives2 = class_representatives2.transpose(0, 1) #tasks_per_batch, n_way, n_support, -1
#     class_representatives2 = class_representatives2.transpose(2, 3).contiguous().view(tasks_per_batch*n_way, -1, n_shot)

#     dist = []
#     for cc in range(tasks_per_batch*n_way):
#         batch_idx = cc//n_way

#         qq1 = query1[batch_idx]
#         qq2 = query2[batch_idx]

#         # get subspace for last fcc output
#         # uu2, ss2, _ = torch.svd(class_representatives2[cc].double())
#         # m2 = torch.sum(~torch.isclose(ss2.float(),torch.tensor(0.0)))
#         # uu2 = uu2.float()[:, :m2]
#         uu2, _, _ = torch.svd(class_representatives2[cc].double())
#         # uu2 = uu2.float()[:, [0]]
#         uu2 = uu2.float()[:, :fl_type[0]]
#         subspace2 = uu2.transpose(0, 1)
        
        

#         # get subspace for second to last fcc output
#         proj_cr = (torch.eye(d).cuda() - uu2.mm(uu2.transpose(0, 1))).mm(class_representatives1[cc])
#         # uu1, ss1, _ = torch.svd(proj_cr.double())
#         # m1 = torch.sum(~torch.isclose(ss1.float(),torch.tensor(0.0)))
#         # uu1 = uu1.float()[:, :m1]
#         uu1, _, _ = torch.svd(proj_cr.double())
#         # uu1 = uu1.float()[:, [0]]
#         uu1 = uu1.float()[:, :fl_type[1]]
#         subspace1 = uu1.transpose(0, 1)


#         projection1 = subspace1.transpose(0, 1).mm(subspace1.mm(qq1.transpose(0, 1))).transpose(0, 1)
#         projection2 = subspace2.transpose(0, 1).mm(subspace2.mm(qq2.transpose(0, 1))).transpose(0, 1)
#         dist_perclass = torch.sum((qq1 - projection1)**2, dim=-1) + torch.sum((qq2 - projection2)**2, dim=-1)
#         dist.append(dist_perclass)

#     dist = torch.stack(dist).view(tasks_per_batch, n_way, -1).transpose(1, 2)
#     logits = -dist

#     if normalize:
#         logits = logits / d
    
#     return logits