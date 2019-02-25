#!/bin/bash

PATH=~/local/bin:/Tools/ScholarlyMarkdown-SLIM/bin:/usr/texbin:/usr/local/bin:/opt/local/bin:/sw/bin:/usr/bin:/bin:/usr/sbin:/sbin
scholdoc --css='https://slimgroup.slim.gatech.edu/ScholMD/standalone/slimweb-scholmd-standalone-v0.1-latest.min.css' --mathjax='https://slimgroup.slim.gatech.edu/MathJax/MathJax.js?config=TeX-AMS_HTML-full' --to=html --default-image-extension=png --citeproc "README_image_proc_constraint_learning.md" "/Tools/ScholarlyMarkdown-SLIM/configs/slim_js.yaml" "/Tools/ScholarlyMarkdown-SLIM/configs/default_csl.yaml" -o "README_image_proc_constraint_learning.html" 2>&1
