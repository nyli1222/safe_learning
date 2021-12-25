import itertools

def batchify(arrays, batch_size):
    """Yield the arrays in batches and in order.
    The last batch might be smaller than batch_size.
    Parameters
    ----------
    arrays : list of ndarray
        The arrays that we want to convert to batches.
    batch_size : int
        The size of each individual batch.
    """
    if not isinstance(arrays, (list, tuple)):
        arrays = (arrays,)

    # Iterate over array in batches
    for i, i_next in zip(itertools.count(start=0, step=batch_size),
                         itertools.count(start=batch_size, step=batch_size)):

        batches = [array[i:i_next] for array in arrays]

        # Break if there are no points left
        if batches[0].size:
            yield i, batches
        else:
            break