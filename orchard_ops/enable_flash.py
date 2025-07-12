# orchard_ops/enable_flash.py

import logging
import os
import sys

def main(verbose=False):
    log_path = os.path.join(os.path.dirname(__file__), "orchard_debug.log")
    # Set up logging to file only, NOT terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        filename=log_path,
        filemode="a"  # append to existing log
    )
    logger = logging.getLogger("orchard_ops.enable_flash")

    try:
        logger.info("[orchard] FlashAttention stub enable_flash invoked (no-op)")
        logger.info(f"[orchard] running from: {os.path.dirname(__file__)}")
        if verbose:
            logger.info("[orchard] (Verbose mode) No actual kernel loaded in stub.")
        # Optional: if you want to record environment/debug info
        logger.info(f"[orchard] PYTHONPATH={os.environ.get('PYTHONPATH')}")
    except Exception as e:
        # Print critical errors to the terminal
        print("[orchard][CRITICAL] Error during logging:", e, file=sys.stderr)
        print(f"[orchard][CRITICAL] See log for details: {log_path}", file=sys.stderr)
        raise

if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    try:
        main(verbose=verbose)
    except Exception as e:
        print(f"[orchard][CRITICAL] Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
