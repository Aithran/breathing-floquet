#!/bin/bash

pip freeze | sort | diff --new-line-format="" --unchanged-line-format="" - <(sort jupyter-requirements.txt) > requirements.txt
