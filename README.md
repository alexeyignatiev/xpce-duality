# XPCE-duality Experiment

A Python script for compiling a classifier into a set of all absolute
explanations or all counterexamples. The setup of the experiment includes the
following files:

* `compile.py` - the compilation script
* `mnist56.lp`, which is an MILP encoding of a ReLU-based NN classifier trained to distinguish digits 5 and 6 in the [MNIST dataset](http://yann.lecun.com/exdb/mnist).
* two text files `sample31.patch` and `sample41.patch` defining two patch areas

## Dependencies

The script requires the following Python packages to be installed first:

* [cplex](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html)
* [matplotlib](https://matplotlib.org/)
* [numpy](http://www.numpy.org/)
* [PySAT](https://github.com/pysathq/pysat)
* [six](https://pythonhosted.org/six/)

Please, follow the installation instructions on these projects' websites to install them properly.

## Usage

The compilation script has a number of parameters, which can be set from the command line. To see the list of options, run:

```
$ compile.py -h
```

### Reproducing the results for digit 5

To reproduce the results shown in the paper and to obtain the corresponding images for digit five, do the following:

#### Compute explanations

```
$ compile.py -p sample31.patch --htype rc2 -d 31 -e 1 -c true -i 2 --save mnist56.lp
```

This command computes all explanations for the classifier using the patch area specified in the file `sample31.patch`, with the remaining pixels being fixed as in the 31st data instance (see option `-d 31`). Option `-e 1` is used to show one concrete explanation with the number 1 (since processing `set` and `dict` may be non-deterministic in Python, you may get a different explanation highlighted).

Besides reporting the pixels participating in each explanation in the terminal, this command will create 4 PNG images showing (1) the original image (`sample31-orig.png`), (2) the image with the patch area highlighted (`sample31-patch.png`), (3) all explanations shown (`sample31-true-all-expls.png`), and (4) one concrete explanation highlighted (`sample31-true-one-expl.png`).

#### Compute counterexamples

```
$ compile.py -p sample31.patch --htype rc2 -d 31 -c opposite -i 2 --save mnist56.lp
```

In contrast to the previous case, this command computes all explanations for the *opposite* prediction, i.e. it computes all counterexamples for the original prediction. It reports all of them in the terminal and additionally creates two PNG images: (1) highlighting all counterexamples (`sample31-opposite-all-expls.png`) and (2) showing one concrete adversarial example constructed from the first counterexample (`sample31-true-ae.png`).

### Reproducing the results for digit 6

Similarly to the case of digit five, the following commands can be used to obtain the results for digit six shown in the paper:

```
$ compile.py -p sample41.patch --htype rc2 -d 41 -e 5 -c true -i 2 --save mnist56.lp
```

and

```
$ compile.py -p sample41.patch --htype rc2 -d 41 -c opposite -i 2 --save mnist56.lp
```

Option `-e 5` in the former command assumes that explanation with index 5 is the one shown in the paper. However, the resulting image (`sample41-true-one-expl.png`) may slightly differ on a different machine. Again, executing these two commands will compute all explanations and counterexamples for the prediction `6` as well as create six images shown in the paper.

### Other data instances

Although this repository gives access to the patch files for the concrete data instances used in the paper (31 and 41), a user may want to play with other data samples. For that, other patch files may be needed. Note that if the script finishes without computing any explanation and any counterexample then this essentially means that the patch area is too small, i.e. the prediction holds no matter what colors of the patch pixels are. In this case, the patch should be be extended to include more pixels.

## Citation

If this work has been significant to a project that leads to an academic publication, please, acknowledge that fact by citing our paper:

```
@inproceedings{inms-neurips19,
  author    = {Alexey Ignatiev and
               Nina Narodytska and
               Joao Marques-Silva},
  title     = {On Relating Explanations and Adversarial Examples},
  booktitle = {NeurIPS},
  year      = {2019}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
