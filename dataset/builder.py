def build_dataloader(cfg):
    if cfg["dataset"] == "mm_imdb":
        from .mm_imdb import build_mmimdb
        return build_mmimdb(cfg)

    elif cfg["dataset"] == "hateful_memes":
        from .hateful_memes import build_memes
        return build_memes(cfg)
