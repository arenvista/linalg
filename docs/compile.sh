#!/bin/bash

# Compile w/ typst compiler 
# Version: typst 0.15.1 (9dfd3a08)

typst compile documentation.typ documentation.pdf
typst compile main.typ main.pdf
