#!/bin/bash

BUCKET="pokeraistore"
PREFIX="preflop/"

echo "🔍 Counting files in s3://${BUCKET}/${PREFIX} ..."

aws s3 ls s3://${BUCKET}/${PREFIX} --recursive \
  | wc -l