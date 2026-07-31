import os
import logging
from datetime import datetime
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

# ---------------------------------------------------------------------------
# Logging setup: logs to console AND to data/logs/scrape_log.txt
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join("data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "scrape_log.txt")

logger = logging.getLogger("scraper")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def count_images(directory):
    if not os.path.isdir(directory):
        return 0
    return len([
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
    ])


def scrape_pokemon_plushies(pokemon_list, images_per_pokemon):
    run_summary = {}

    for pokemon in pokemon_list:
        logger.info(f"=== Starting download for: {pokemon} ===")

        # Target directory: data/raw/<pokemon_name>
        output_dir = os.path.join("data", "raw", pokemon.lower())
        os.makedirs(output_dir, exist_ok=True)

        queries = [
            f"{pokemon} plush",
            f"{pokemon} plushie",
            f"{pokemon} plushie doll",
            f"{pokemon} stuffed animal",
            f"{pokemon} stuffed toy",
        ]

        # Split the budget across both engines and all queries.
        per_query_count = max(1, images_per_pokemon // (len(queries) * 2))

        pokemon_start_count = count_images(output_dir)

        for query in queries:
            # --- Bing ---
            before = count_images(output_dir)
            logger.info(f"  [Bing]   searching: '{query}' (target {per_query_count})")
            try:
                bing_crawler = BingImageCrawler(
                    downloader_threads=4,
                    storage={"root_dir": output_dir},
                )
                bing_crawler.crawl(
                    keyword=query,
                    max_num=per_query_count,
                    file_idx_offset="auto",
                )
            except Exception as e:
                logger.warning(f"  [Bing]   FAILED on '{query}': {e}")
            after = count_images(output_dir)
            logger.info(f"  [Bing]   '{query}' -> +{after - before} images (total: {after})")

            # --- Google ---
            before = count_images(output_dir)
            logger.info(f"  [Google] searching: '{query}' (target {per_query_count})")
            try:
                google_crawler = GoogleImageCrawler(
                    downloader_threads=4,
                    storage={"root_dir": output_dir},
                )
                google_crawler.crawl(
                    keyword=query,
                    max_num=per_query_count,
                    file_idx_offset="auto",
                )
            except Exception as e:
                logger.warning(f"  [Google] FAILED on '{query}': {e}")
            after = count_images(output_dir)
            logger.info(f"  [Google] '{query}' -> +{after - before} images (total: {after})")

        pokemon_end_count = count_images(output_dir)
        gained = pokemon_end_count - pokemon_start_count
        run_summary[pokemon] = {
            "start": pokemon_start_count,
            "end": pokemon_end_count,
            "gained": gained,
        }
        logger.info(f"=== Finished {pokemon}: {gained} new images (total in folder: {pokemon_end_count}) ===\n")

    logger.info("========== RUN SUMMARY ==========")
    for pokemon, stats in run_summary.items():
        logger.info(f"{pokemon:20s} +{stats['gained']:4d} new  | folder total: {stats['end']}")
    logger.info("==================================")

    return run_summary


if __name__ == "__main__":

    target_pokemon = ["munchlax", "pikachu", "bulbasaur", "charmander", "squirtle", "gengar"]
    scrape_pokemon_plushies(target_pokemon, 100)