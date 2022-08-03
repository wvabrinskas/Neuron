#!/bin/zsh

swift package --allow-writing-to-directory ../docs \
    generate-documentation --target Neuron \
    --disable-indexing \
    --transform-for-static-hosting \
    --hosting-base-path Neuron \
    --output-path ../docs