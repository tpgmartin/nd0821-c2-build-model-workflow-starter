#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import os
import pandas as pd
import tempfile
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    logger.info("Downloading and reading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path, low_memory=False)

    # Split first in model_dev/test, then we further divide model_dev in train and validation
    logger.info("Drop outliers and convert `last_review` to datetime")
    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Save the artifacts. 
    # We use a temporary directory so we do not leave any trace behind
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Get the path on disk within the temp directory
        temp_path = os.path.join(tmp_dir, args.output_artifact)

        logger.info(f"Uploading dataset to {args.output_artifact}")

        # Save then upload to W&B
        df.to_csv(temp_path, index=False)

        artifact = wandb.Artifact(
            name=args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )
        artifact.add_file(temp_path)

        logger.info("Logging artifact")
        run.log_artifact(artifact)

        artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of input dataset",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of output dataset",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Datatype of output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price cutoff",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price cutoff",
        required=True
    )


    args = parser.parse_args()

    go(args)
