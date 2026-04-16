import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run all scrapers for a specific time frame.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    scripts = [
        "scrapers/scrape_finance.py",
        "scrapers/scrape_geo.py",
        "scrapers/scrape_news.py",
        "scrapers/fetch_prices.py"
    ]

    print(f"============================================================")
    print(f" Starting All Scrapers | Window: {args.start} to {args.end}")
    print(f"============================================================\n")

    for script in scripts:
        print(f">>> Running {script}...")
        try:
            # We use run and check=True so it raises an error if the script fails.
            subprocess.run([
                sys.executable, script, 
                "--start", args.start, 
                "--end", args.end
            ], check=True)
            print(f">>> Finished {script}\n")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running {script}. Exiting.")
            sys.exit(1)

    print("============================================================")
    print(" All scraping completed successfully!")
    print("============================================================")

if __name__ == "__main__":
    main()
