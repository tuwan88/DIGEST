import torch
from torch_geometric.nn.pool import global_add_pool,global_mean_pool

def mapping_loss(pred_mapping_label,data):
    mapping_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    n1 = data.n[:,0:1]
    n2 = data.n[:,1:]
    mapping_batch = data.batch[data.edge_index_mapping[0]]
    epoch_percent = 0.5
    if epoch_percent >= 1.0:
        loss = mapping_loss(pred_mapping_label, data.edge_attr_mapping)
        reduce_loss = global_mean_pool(loss,mapping_batch).sum()
        return reduce_loss

    num_1 = global_add_pool(data.edge_attr_mapping,mapping_batch)
    
    num_0 = n1 * n2 - num_1
    
    mask_1 = (num_1 >= num_0)[mapping_batch]
    
    p_base = num_1 / num_0
    p = 1.0 - (p_base + epoch_percent * (1-p_base))
    
    mask_2 = (torch.rand_like(data.edge_attr_mapping).to(data.edge_attr_mapping.device) + data.edge_attr_mapping) > p[mapping_batch]

    loss_mask = (mask_1 | mask_2).squeeze(1)

    loss = mapping_loss(pred_mapping_label[loss_mask], data.edge_attr_mapping[loss_mask])
    reduce_loss = global_mean_pool(loss,mapping_batch[loss_mask]).sum()
    
    return reduce_loss