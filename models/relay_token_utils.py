"""
Utils for processing multi-scale relay tokens in batches.

Adapted by Ethan Griffiths (Data61, Pullenvale)
"""
import torch
from torch.nn.utils.rnn import unpad_sequence

from typing import Optional, List, Dict, Union
from models.octree import OctreeT, pad_sequence

def concat_and_pad_rt(
    relay_token_dict: Dict[int, torch.Tensor],
    octree: OctreeT,
    pyramid_depths: Optional[List[int]] = None,
    pad: bool = True,
    remove_final_padding: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Concatenates relay tokens from different levels in the pyramid
    batch-wise, then applies padding for parallelisation. Returns a single
    (B, N, C) tensor, where B = batch size, N = number of tokens (incl.
    padding), and C = channel size. Optionally allows returning the batch list
    of relay tokens without padding.
    NOTE: If remove_final_padding is True, the padded relay tokens from the final
    batch element will be removed. This returns the list of real relay tokens for
    each batch element (for processing after the end of the network), but will
    cause issues if you try to call `unpad_and_split_rt` on it.
    """
    assert not (pad and remove_final_padding), '`pad` must be False if `remove_final_padding` is True'
    if pyramid_depths is None:
        pyramid_depths = octree.pyramid_depths
    # Split relay tokens into batches for each depth
    relay_tokens_split_per_depth = []
    for depth_j in pyramid_depths:
        batch_num_relay_tokens_depth_j = octree.batch_num_windows[depth_j].tolist()
        relay_tokens_split_per_depth.append(
            list(relay_token_dict[depth_j].split(batch_num_relay_tokens_depth_j))
        )
        if remove_final_padding:  # Remove padded RTs from end of batch
            rt_batch_idx_list = octree.ct_batch_idx[depth_j].split(batch_num_relay_tokens_depth_j)
            final_batch_num_relay_tokens = torch.count_nonzero(rt_batch_idx_list[-1] == len(rt_batch_idx_list) - 1)
            relay_tokens_split_per_depth[-1][-1] = relay_tokens_split_per_depth[-1][-1][:final_batch_num_relay_tokens]
    
    # Combine relay tokens for each batch in all depths
    relay_tokens_combined_list = []
    for relay_token_pyramid_batch in zip(*relay_tokens_split_per_depth):
        relay_tokens_combined_list.append(
            torch.cat(relay_token_pyramid_batch)
        )
    if not pad:
        return relay_tokens_combined_list
    padded_pyramid_relay_tokens = pad_sequence(relay_tokens_combined_list)
    return padded_pyramid_relay_tokens

def unpad_and_split_rt(
    pyramid_relay_tokens_list_or_padded_tensor: Union[torch.Tensor, List[torch.Tensor]],
    octree: OctreeT,
    pyramid_depths: Optional[List[int]] = None,
) -> dict[int, torch.Tensor]:
    """
    Reverses the concatenation and padding applied to multi-scale relay
    tokens. Returns a dictionary where keys are octree depth, and values
    are the corresponding relay tokens in a (M, C) tensor. Also accepts a 
    list of each batch of relay tokens without padding.
    """
    if pyramid_depths is None:
        pyramid_depths = octree.pyramid_depths
    # Remove padding
    if isinstance(pyramid_relay_tokens_list_or_padded_tensor, torch.Tensor):
        relay_tokens_combined_list = unpad_sequence(
            pyramid_relay_tokens_list_or_padded_tensor,
            octree.batch_num_relay_tokens_combined,
            batch_first=True
        )
    elif isinstance(pyramid_relay_tokens_list_or_padded_tensor, list):
        relay_tokens_combined_list = pyramid_relay_tokens_list_or_padded_tensor
    else:
        raise ValueError
    # Separate relay tokens for each depth
    batch_num_relay_tokens_per_depth = [
        octree.batch_num_windows[depth_j].tolist()
            for depth_j in pyramid_depths
    ]
    relay_tokens_split_per_depth = [
        [] for _ in range(len(pyramid_depths))
    ]
    for i, batch_num_tokens in enumerate(zip(*batch_num_relay_tokens_per_depth)):
        relay_tokens_split_temp = relay_tokens_combined_list[i].split(
            batch_num_tokens
        )
        for j in range(len(pyramid_depths)):
            relay_tokens_split_per_depth[j].append(
                relay_tokens_split_temp[j]
            )

    # Concatenate relay tokens for each depth and put back in dict
    relay_token_dict = {}
    for i, depth_j in enumerate(pyramid_depths):
        relay_token_dict[depth_j] = torch.cat(relay_tokens_split_per_depth[i])
    
    return relay_token_dict