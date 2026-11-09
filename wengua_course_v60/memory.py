"""Release inactive cache between finished work; never change tensor precision."""
def release_inactive_cuda(*,force=False):
    import torch
    if not torch.cuda.is_available():return None
    free,total=torch.cuda.mem_get_info();reserved=torch.cuda.memory_reserved();allocated=torch.cuda.memory_allocated()
    if not force and (free>=2**30 or reserved-allocated<2**30):return None
    torch.cuda.empty_cache();after,_=torch.cuda.mem_get_info()
    return {'before_free_gib':free/2**30,'after_free_gib':after/2**30,'before_reserved_gib':reserved/2**30,'after_reserved_gib':torch.cuda.memory_reserved()/2**30,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'tensor_precision_changed':False}


def telemetry():
    import torch
    stats=torch.cuda.memory_stats()
    return {'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'allocator_retries':stats.get('num_alloc_retries'),'allocator_ooms':stats.get('num_ooms')}
