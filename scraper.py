import os
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

def scrape_pokemon_plushies(pokemon_list, images_per_pokemon=60):
    for pokemon in pokemon_list:
        print(f"\n[+] Starting download for: {pokemon}")
        
        # Target directory: data/raw/<pokemon_name>
        output_dir = os.path.join("data", "raw", pokemon.lower())
        os.makedirs(output_dir, exist_ok=True)
        
     
        queries = [
            f"{pokemon} plush",
            f"{pokemon} plushie doll",
            f"{pokemon} stuffed animal"
        ]
        
       
        per_query_count = images_per_pokemon // len(queries)
        
        for query in queries:
            print(f"  --> Searching Bing for: '{query}'")
            bing_crawler = BingImageCrawler(
                downloader_threads=4,
                storage={'root_dir': output_dir}
            )
            bing_crawler.crawl(
                keyword=query,
                max_num=per_query_count,
                file_idx_offset='auto'
            )

if __name__ == "__main__":

    target_pokemon = ["pikachu", "charmander", "squirtle", "bulbasaur", "gengar"]
    scrape_pokemon_plushies(target_pokemon, images_per_pokemon=60)