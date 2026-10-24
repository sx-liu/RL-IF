import logging
import os

def get_logger(name: str = "rl_attribution", log_dir: str = None, filename: str = "run.log"):
    """
    Simple logger factory.
    Logs both to console and optionally to a file under `log_dir`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    _orig_info = logger.info
    def patched_info(*args, **kwargs):

        msg = " ".join(str(a) for a in args)
        return _orig_info(msg, **kwargs)
    logger.info = patched_info

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, filename))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
