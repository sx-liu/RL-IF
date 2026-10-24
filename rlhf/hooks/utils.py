import torch
from typing import Dict, List, Optional, Callable
from torch import Tensor


def _rademacher(
    matrix: Tensor,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> Tensor:
    """Generate Rademacher random matrix in-place.

    Args:
        matrix (Tensor): The matrix to fill with Rademacher values.
        generator (torch.Generator): Random number generator.
        dtype (torch.dtype): Target dtype for the matrix.

    Returns:
        Tensor: The filled matrix.
    """
    matrix.bernoulli_(p=0.5, generator=generator)
    matrix *= 2.0
    matrix -= 1.0
    return matrix.to(dtype=dtype)


def _mask(
    input_dim: int,
    output_dim: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """Generate sorted random mask indices.

    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension (number of indices to select).
        generator (torch.Generator): Random number generator.
        device (torch.device): Device for the indices.

    Returns:
        Tensor: Sorted indices tensor.
    """
    indices = torch.randperm(
        input_dim,
        generator=generator,
        device=device,
    )[:output_dim]
    return indices.sort()[0]

def apply_hadamard_matvec(
    M_proj_forward: Callable,
    source_vector: Tensor,
    result_size: int,
    chunk_size: int,
    source_size: int,
    device: torch.device,
    dtype: torch.dtype
) -> Tensor:
    """
    Compute (M ⊙ M) @ source_vector efficiently using chunked basis vectors.

    This is a helper for computing v_new = (M ⊙ M) @ v where M is a projection matrix.
    Instead of materializing M, we compute it chunk by chunk using basis vectors.

    Args:
        M_proj_forward: Function that maps basis vectors to get chunks of M^T
                       (e.g., projector.forward or composition of transpose+forward)
        source_vector: Vector to multiply [source_size]
        result_size: Size of output vector
        chunk_size: Number of basis vectors per chunk
        source_size: Size of source vector (dimension to chunk over)
        device: Device for computation
        dtype: Data type for computation

    Returns:
        Result vector [result_size]
    """
    v_new = torch.zeros(result_size, device=device, dtype=dtype)
    num_chunks = (source_size + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, source_size)
        chunk_size_actual = chunk_end - chunk_start

        # Create identity chunk representing basis vectors
        I_chunk = torch.zeros(chunk_size_actual, source_size, device=device, dtype=dtype)
        for i in range(chunk_size_actual):
            I_chunk[i, chunk_start + i] = 1.0

        # Map through projector to get M_chunk^T
        M_chunk_T = M_proj_forward(I_chunk)  # [chunk_size, result_size]

        # Transpose and compute Hadamard square
        M_chunk = M_chunk_T.T  # [result_size, chunk_size]
        M_sq_chunk = M_chunk.square()  # [result_size, chunk_size]

        # Accumulate: v_new += (M_chunk ⊙ M_chunk) @ source_vector[chunk]
        source_chunk = source_vector[chunk_start:chunk_end]  # [chunk_size]
        v_new.add_(torch.mv(M_sq_chunk, source_chunk))

    return v_new

def stochastic_diagonal_estimation(
    compute_Mz_func: Callable[[Tensor], Tensor],
    result_size: int,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42
) -> Tensor:
    """
    Estimate diagonal of matrix A using stochastic estimation.

    For a matrix A, computes diag(A) ≈ (1/N) Σ z_i ⊙ (A z_i) where z_i are Rademacher vectors.

    Args:
        compute_Mz_func: Function that computes A @ z given z [1, k] -> [1, k]
        result_size: Size of the diagonal vector to estimate
        num_samples: Number of Rademacher samples to use
        device: Device for computation
        dtype: Data type for computation
        seed: Random seed for reproducibility

    Returns:
        Estimated diagonal vector [result_size]
    """
    # Initialize accumulator
    v_accum = torch.zeros(result_size, device=device, dtype=dtype)

    # Set random seed for reproducibility
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Stochastic sampling loop
    for sample_idx in range(num_samples):
        # 1. Sample z ~ Rademacher({-1, 1}) [1, k]
        z = torch.randint(0, 2, (1, result_size),
                        generator=generator,
                        device=device, dtype=dtype) * 2 - 1

        # 2. Compute y_i = A @ z_i
        y_i = compute_Mz_func(z)  # [1, k]

        # 3. Compute h_i = z_i ⊙ y_i (element-wise product)
        h_i = z * y_i  # [1, k]

        # 4. Accumulate
        v_accum.add_(h_i.squeeze(0))

    # Average over samples
    v_result = v_accum / num_samples

    return v_result

def vectorize(
    g: Dict[str, Tensor],
    batch_dim: Optional[bool] = True,
    arr: Optional[Tensor] = None,
    device: Optional[str] = "cuda",
) -> Tensor:
    """
    Vectorize gradients into a flattened tensor.

    This function takes a dictionary of gradients and returns a flattened tensor
    of shape [batch_size, num_params].

    Args:
        g: A dictionary containing gradient tensors to be vectorized
        batch_dim: Whether to include the batch dimension in the returned tensor
        arr: An optional pre-allocated tensor to store the vectorized gradients
        device: The device to store the tensor on

    Returns:
        A flattened tensor of gradients
    """
    if arr is None:
        if batch_dim:
            g_elt = g[next(iter(g.keys()))]
            batch_size = g_elt.shape[0]
            num_params = 0
            for param in g.values():
                if param.shape[0] != batch_size:
                    msg = "Parameter row num doesn't match batch size."
                    raise ValueError(msg)
                num_params += int(param.numel() / batch_size)
            arr = torch.empty(
                size=(batch_size, num_params),
                dtype=g_elt.dtype,
                device=device,
            )
        else:
            num_params = 0
            for param in g.values():
                num_params += int(param.numel())
            arr = torch.empty(size=(num_params,), dtype=param.dtype, device=device)

    pointer = 0
    vector_dim = 1
    for param in g.values():
        if batch_dim:
            if len(param.shape) <= vector_dim:
                num_param = 1
                p = param.data.reshape(-1, 1)
            else:
                num_param = param[0].numel()
                p = param.flatten(start_dim=1).data
            arr[:, pointer : pointer + num_param] = p.to(device)
            pointer += num_param
        else:
            num_param = param.numel()
            arr[pointer : pointer + num_param] = param.reshape(-1).to(device)
            pointer += num_param
    return arr

def get_parameter_chunk_sizes(
    param_shape_list: List,
    batch_size: int,
) -> tuple[int, List[int]]:
    """Compute chunk size information from feature to be projected.

    Get a tuple containing max chunk size and a list of the number of
    parameters in each chunk.

    Args:
        param_shape_list (List): A list of numbers indicating the total number of
            features to be projected. A typical example is a list of parameter
            size of each module in a torch.nn.Module model.
        batch_size (int): The batch size. Each term (or module) in feature
            will have the same batch size.

    Returns:
        tuple[int, List[int]]: A tuple containing:
            - Maximum number of parameters per chunk
            - A list of the number of parameters in each chunk
    """
    param_shapes = torch.tensor(param_shape_list, dtype=torch.int64)

    max_chunk_size = torch.iinfo(torch.int32).max // (batch_size * 8)

    params_per_chunk = []
    chunk_sum = 0

    for ps in param_shapes:
        # If adding the current param exceeds the max size,
        # finalize the current chunk (if not empty) and start a new one.

        current_ps = ps

        if chunk_sum + current_ps >= max_chunk_size:
            if chunk_sum > 0:
                params_per_chunk.append(chunk_sum)
            chunk_sum = 0  # Reset for new chunk

        # Handle the case where a single param layer is
        # larger than the max_chunk_size by splitting it.
        while current_ps >= max_chunk_size:
            params_per_chunk.append(max_chunk_size)
            current_ps -= max_chunk_size

        # Add the (remainder of) the current param to the chunk
        chunk_sum += current_ps

    # Add the final chunk if it has any params
    if chunk_sum > 0:
        params_per_chunk.append(chunk_sum)

    # Handle edge case of no params
    if not params_per_chunk:
        params_per_chunk = [0]

    return max_chunk_size, params_per_chunk

def create_sample_inputs(
    tokenizer,
    max_seq_length: int = 512,
    device: str = 'cpu',
    sample_text: Optional[str] = None
) -> Dict[str, Tensor]:
    """
    Create sample inputs for initializing gradient compressors.

    This helper function creates a minimal batch of tokenized inputs that can be used
    to run a forward pass through the model to determine layer dimensions.

    Args:
        tokenizer: HuggingFace tokenizer instance
        max_seq_length: Maximum sequence length for tokenization
        device: Device to place the tensors on
        sample_text: Optional sample text to tokenize. If None, uses a default message.

    Returns:
        Dictionary containing tokenized inputs ready for model forward pass

    Example:
        >>> sample_inputs = create_sample_inputs(tokenizer, max_seq_length=512, device='cuda')
        >>> sparsifiers, projectors = setup_model_compressors(
        ...     model=model,
        ...     layer_names=layer_names,
        ...     sample_inputs=sample_inputs,
        ...     device='cuda'
        ... )
    """
    if sample_text is None:
        sample_text = "This is a sample text for compression initialization."

    sample_inputs = tokenizer(
        [sample_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length
    )

    # Move to device and add labels for language modeling
    sample_inputs = {k: v.to(device) for k, v in sample_inputs.items()}
    sample_inputs['labels'] = sample_inputs['input_ids'].clone()

    return sample_inputs

def topk_selection(scores, k: int):
    """
    Curate k data points with the highest scores using simple top-k.

    This is O(n + k log k) and much faster than greedy_selection.
    Use this when second-order interactions are not needed.

    Arguments:
    - scores: A tensor of scores for each data point.
    - k: The number of data points to select.

    Returns:
    - selected_indices: Tensor of indices of the selected data points (on same device as scores).
    """
    # Clamp k to avoid selecting more than available samples (e.g., last batch may be smaller)
    k = min(k, scores.shape[0])
    _, selected_indices = torch.topk(scores, k, largest=True, sorted=False)
    return selected_indices


def negative_filtering(scores, filter_frac: float = 1.0):
    """
    Filter out negative-influence samples.

    Note that it is possible to return empty result when all samples have
    negative influence.

    Arguments:
    - scores: A tensor of influence scores for each data point.
    - filter_frac: Fraction of negative-influence samples to drop (0.0 to 1.0).
                   - 0.0: Keep all samples (no filtering)
                   - 1.0: Drop all negative-influence samples

    Returns:
    - selected_indices: Tensor of indices of the kept data points (on same device as scores).
                       May be empty if all samples have negative influence.
    """
    n_samples = scores.shape[0]
    device = scores.device

    # Handle edge case: no filtering
    if filter_frac <= 0.0:
        return torch.arange(n_samples, device=device)

    # Fast path: filter_frac >= 1.0 means drop ALL negative samples.
    # Single kernel: just find non-negative indices.
    if filter_frac >= 1.0:
        return (scores >= 0.0).nonzero(as_tuple=False).view(-1)

    # General case: drop bottom filter_frac of negative samples
    positive_mask = scores >= 0.0
    negative_mask = ~positive_mask

    positive_indices = positive_mask.nonzero(as_tuple=False).view(-1)
    negative_indices = negative_mask.nonzero(as_tuple=False).view(-1)
    n_negative = negative_indices.shape[0]

    if n_negative == 0:
        return positive_indices

    n_to_drop = int(n_negative * filter_frac)

    if n_to_drop >= n_negative:
        return positive_indices

    # Keep the least-negative ones: sort descending, keep top (n_neg - n_to_drop)
    negative_scores = scores[negative_indices]
    sorted_neg_indices = torch.argsort(negative_scores, descending=True)
    n_to_keep = n_negative - n_to_drop
    kept_negative_indices = negative_indices[sorted_neg_indices[:n_to_keep]]

    return torch.cat([positive_indices, kept_negative_indices])


def greedy_selection(scores, interaction_matrix, k: int):
    """
    Curate k data points based on the highest scores, dynamically updating scores
    by subtracting interactions with previously selected data points.

    This is O(k*n) and accounts for second-order interactions between samples.
    For faster curation without interactions, use topk_selection instead.

    Arguments:
    - scores: A tensor of initial scores for each data point.
    - interaction_matrix: A tensor of pairwise interactions between data points.
    - k: The number of data points to select.

    Returns:
    - selected_indices: Tensor of indices of the selected data points (on same device as scores).
    """
    # Clamp k to avoid selecting more than available samples (e.g., last batch may be smaller)
    k = min(k, scores.shape[0])
    selected_indices = torch.empty(k, dtype=torch.long, device=scores.device)
    scores = scores.clone()

    for i in range(k):
        idx_max = torch.argmax(scores)
        selected_indices[i] = idx_max

        # Update scores by subtracting interactions with the selected data point
        scores -= interaction_matrix[idx_max, :]
        scores[idx_max] = -float('inf')

    return selected_indices