# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


import shutil
from pathlib import Path

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """Process raw data into processed data."""
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_path = Path(input_filepath)
    output_path = Path(output_filepath)

    # create processed directory
    output_path.mkdir(parents=True, exist_ok=True)

    # copy all csv files (baseline)
    for file in input_path.glob("*.csv"):
        shutil.copy(file, output_path / file.name)

    logger.info("Processed data saved to %s", output_path)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
