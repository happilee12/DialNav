
def init_distributed(opts):
    init_param = load_init_param(opts)
    rank = init_param["rank"]

    print(f"Init distributed {init_param['rank']} - {init_param['world_size']}")

    dist.init_process_group(**init_param)
    return rank


def is_default_gpu(opts) -> bool:
    print("opts", opts)
    print(opts.local_rank)
    return opts.local_rank == -1 or dist.get_rank() == 0