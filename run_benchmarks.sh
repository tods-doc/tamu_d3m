#!/bin/bash -e

if ! git remote get-url upstream > /dev/null 2>&1 ; then
  git remote add upstream https://gitlab.com/datadrivendiscovery/d3m.git
fi
git fetch upstream

asv machine --yes --config tests/asv.conf.json

echo ""

if asv continuous upstream/devel HEAD --split --factor 1.1 --show-stderr --config tests/asv.conf.json ; then
  echo "Benchmarks ran without errors."
else
  echo "Benchmarks have errors."
fi
