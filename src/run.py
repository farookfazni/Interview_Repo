from datasets import load_dataset
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, lower, count
import datetime
import logging
from typing import List
import argparse

# all in one function
def process_data(input_path: str, output_path: str, words_to_count: List[str]) -> None:
    """
    Processes a dataset by loading it, converting it to Parquet format, and counting occurrences of specific words.
    
    Args:
        input_path (str): Path to the dataset. should be a hugging Face dataset name
        output_path (str): Directory to save the output Parquet file.
        words_to_count (List[str]): List of words to count occurrences of.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Loading dataset...")
    ag_news_dataset = load_dataset(input_path)
    
    logging.info("Converting dataset to Parquet format...")
    ag_news_dataset['train'].to_parquet('train.parquet')
    
    logging.info("Initializing Spark session...")
    spark = SparkSession.builder.appName('WordCount').getOrCreate()
    
    logging.info("Loading Parquet file into Spark DataFrame...")
    df = spark.read.parquet("train.parquet")
    
    YYYYMMDD = datetime.datetime.now().strftime('%Y%m%d')
    
    logging.info("Counting occurrences of specific words...")
    result = df.select(explode(split(lower(col('description')), ' ')).alias('word')) \
                .filter(col('word').isin(words_to_count)) \
                .groupBy('word') \
                .agg(count('*').alias('word_count'))
    
    output_file = f"{output_path}/word_count_{YYYYMMDD}.parquet"
    
    logging.info(f"Saving results to {output_file}...")
    result.write.mode('overwrite').parquet(output_file)
    
    logging.info("Process completed successfully.")

def process_data_all(input_path: str, output_path: str) -> None:
    """
    Processes a dataset by loading it, converting it to Parquet format, and counting occurrences of all words.
    
    Args:
        input_path (str): Path to the dataset. should be a hugging Face dataset name
        output_path (str): Directory to save the output Parquet file.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Loading dataset...")
    ag_news_dataset = load_dataset(input_path)
    
    logging.info("Converting dataset to Parquet format...")
    ag_news_dataset['train'].to_parquet('train.parquet')
    
    logging.info("Initializing Spark session...")
    spark = SparkSession.builder.appName('WordCountAll').getOrCreate()
    
    logging.info("Loading Parquet file into Spark DataFrame...")
    df = spark.read.parquet("train.parquet")
    
    YYYYMMDD = datetime.datetime.now().strftime('%Y%m%d')
    
    logging.info("Counting occurrences of all words...")
    result = df.select(explode(split(lower(col('description')), ' ')).alias('word')) \
                .groupBy('word') \
                .agg(count('*').alias('word_count'))
    
    output_file = f"{output_path}/word_count_{YYYYMMDD}.parquet"
    
    logging.info(f"Saving results to {output_file}...")
    result.write.mode('overwrite').parquet(output_file)
    
    logging.info("Process completed successfully.")


# words_to_count = ["president", "the", "Asia"]

# process_data('sh0416/ag_news','output',words_to_count)

# process_data_all('sh0416/ag_news','output/all_words')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data to generate word frequency tables.")
    parser.add_argument("command", choices=["process_data", "process_data_all"], help="Command to execute")
    # parser.add_argument("-cfg", type=str, required=True, help="Path to the config file")
    parser.add_argument("-dataset", type=str, required=True, help="Dataset to process")
    parser.add_argument("-dirout", type=str, required=True, help="Output directory")
    parser.add_argument("-words", nargs="*", help="Specific words to count (for process_data)", default=["president", "the", "Asia"])

    args = parser.parse_args()

    if args.command == "process_data":
        process_data(args.dataset, args.dirout, args.words)

    elif args.command == "process_data_all":
        process_data_all(args.dataset, args.dirout)
