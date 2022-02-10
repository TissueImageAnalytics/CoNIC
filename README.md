# Docker Templates for CoNIC

In this repository, you can find two docker container templates for [CoNIC challenge](https://conic-challenge.grand-challenge.org/Home/):

- `conic_template_blank`: This directory contains a docker template with all the essential functions and modules needed to create an acceptable docker container for the CoNIC challenge. Almost all of the functions and instructions in this template should remain the same and you just need to add/link your algorithm and weight files to them.
- `conic_template_baseline`: This directory contains a sample algorithm ready to be dockerized for the CoNIC challenge. In other words, `conic_template_baseline` contains all necessary files available in the `conic_template_blank` as well as some other python functions and modules related to the [CoNIC baseline method](https://github.com/vqdang/hover_net/tree/conic).

Each of these directories is accompanied by a `README.md` file in which we have thoroughly explained how you can dockerize your algorithms and submit them to the challenge. The code in the `conic_template_blank` has been extensively commented and users should be able to embed their algorithms in the blank template, however, `conic_template_baseline` can be a good guide (example) to better understand the acceptable algorithm layout. 

For more information, please have a look at our [tutorial videos](https://conic-challenge.grand-challenge.org/Videos/).
