#!/usr/bin/env python
#-*- coding:utf-8 -*-


from __future__ import print_function
import cplex
import getopt
import math
import matplotlib.image as mpimg
import matplotlib.cm as mpcm
import numpy as np
import os
from pysat.card import CardEnc, EncType
from pysat.examples.hitman import Hitman
import resource
from six.moves import range
import sys


def parse_lpfile(fname):
    """
        Parse LP file and extract LP constraints as well as variable names and
        the original dataset.
    """

    with open(fname, 'r') as fp:
        file_content = fp.readlines()

    data = []

    for line in file_content:
        if line[0] != '\\':
            break
        elif line.startswith('\\ data:'):
            data.append(line[7:].strip())
        elif line.startswith('\\ inputs:'):
            inputs = list(line[10:].strip().split(', '))
        elif line.startswith('\\ outputs:'):
            outputs = list(line[11:].strip().split(', '))

    ilp = cplex.Cplex()
    ilp.read(fname)

    return ilp, inputs, outputs, data


def prepare_hitman(pixels, inputs, intervals, htype):
    """
        Initialize a hitting set enumerator.
    """

    if not pixels:
        pixels = sorted(range(len(inputs)))

    # new Hitman object
    h = Hitman(htype=htype)

    # first variables should be related with the elements of the sets to hit
    # that is why we are adding soft clauses first
    for p in pixels:
        for v in range(intervals):
            var = h.idpool.id(tuple([inputs[p], v]))
            h.oracle.add_clause([-var], 1)

    # at most one value per pixel can be selected
    for p in pixels:
        lits = [h.idpool.id(tuple([inputs[p], v])) for v in range(intervals)]
        cnf = CardEnc.atmost(lits, encoding=EncType.pairwise)

        for cl in cnf.clauses:
            h.oracle.add_clause(cl)

    return h


def compile_classifier(h, x, inputs, pmap, intervals, verb):
    """
        Compile the classifier using ILP.
    """

    expls, coexs = [], []
    hset = []

    while True:
        if hset:
            print('hset', ', '.join(['(p{0}, {1})'.format(pmap[l[0]], l[1]) for l in hset]))
        else:
            print('hset -')

        # check if the hitting set is an explanation, i.e. there
        # is no instance matching it but classified differently
        coex = x.get_counterexample(assumptions=hset)

        if coex:
            # the opposite class
            # an explanation needs to be extracted
            expl = x.explain(coex)
            coexs.append(expl)

            hit_counterexample(coex, expl, h, inputs, intervals)

            print('coex', ', '.join(['(p{0}, {1})'.format(pmap[inputs[l[0]]], int(coex[l[0]])) for l in expl]))
        else:
            # if the class is 0 then no reduction is needed
            # the hitting set should be a subset-minimal explanation
            h.block(hset)
            expls.append(hset)

            print('expl', ', '.join(['(p{0}, {1})'.format(pmap[l[0]], l[1]) for l in hset]))

        # new hitting set
        hset = h.get()
        if hset == None:
            break

        hset = list(filter(lambda x: len(x) == 2, hset))
        print('')

    return expls, coexs


def hit_counterexample(coex, expl, h, inputs, intervals):
    """
        Encode the negation of the counterexample so that it will be hit next
        time.
    """

    encoded = []

    for l in expl:
        pix, val = inputs[l[0]], int(coex[l[0]])
        lobj = tuple([pix, val, '-'])

        if lobj not in h.idpool.obj2id:
            v = h.idpool.id(lobj)  # new variable identifier

            # pixel cannot take value v and not v simultaneously
            h.oracle.add_clause([-v, -h.idpool.id(tuple([pix, val]))])

            # v implies that some other value must be chosen for this pixel
            cl = []
            for i in set(range(intervals)).difference(set([val])):
                v2 = h.idpool.id(tuple([inputs[l[0]], i]))
                cl.append(v2)

                # this clause is optional
                h.oracle.add_clause([-v2, v])

            h.oracle.add_clause([-v] + cl)
        else:
            v = h.idpool.id(lobj)

        encoded.append(v)

    h.oracle.add_clause(encoded)


class ILPExplainer(object):
    """
        An ILP-inspired minimal explanation extractor for neural networks
        based on ReLUs.
    """

    def __init__(self, oracle, inputs, outputs, names, datainst, free,
            compile_, verbose=0):
        """
            Constructor.
        """

        self.verbose = verbose
        self.oracle = oracle

        # turning logger off
        self.oracle.set_log_stream(None)
        self.oracle.set_error_stream(None)
        self.oracle.set_results_stream(None)
        self.oracle.set_warning_stream(None)

        # feature and class names (for verbosity)
        self.fnames = names

        # internal input variable names
        self.inputs = inputs

        # output variable names
        self.outputs = outputs

        # true class of the counterexample
        self.coex_class = None

        # free inputs (by default, include all inputs)
        if free:
            self.free = set(free)
        else:
            self.free = set(range(len(self.inputs)))

        # now, we need to freeze the non-free inputs
        if len(self.free) < len(inputs):
            for i, (var, val) in enumerate(zip(inputs, datainst)):
                if i in self.free:
                    continue

                eql, rhs = [[var], [1]], [val]

                cnames = ['freezed_{0}'.format(i)]
                senses = ['E']
                constr = [eql]

                self.oracle.linear_constraints.add(lin_expr=constr,
                        senses=senses, rhs=rhs, names=cnames)

        # hypotheses
        self.hypos = []

        # adding indicators for correct and wrong outputs
        self.oracle.variables.add(names=['c_{0}'.format(i) for i in range(len(outputs))], types='B' * len(outputs))
        for i in range(len(outputs)):
            ivar = 'c_{0}'.format(i)
            wrong = ['wc_{0}_{1}'.format(i, j) for j in range(len(outputs)) if i != j]
            self.oracle.variables.add(names=wrong, types='B' * len(wrong))

            # ivar implies at least one wrong class
            self.oracle.indicator_constraints.add(indvar=ivar, lin_expr=[wrong,
                [1] * len(wrong)], sense='G', rhs=1)

            for j in range(len(outputs)):
                if i != j:
                    # iv => (o_j - o_i >= 0.0000001)

                    iv = 'wc_{0}_{1}'.format(i, j)
                    ov, oc = [outputs[j], outputs[i]], [1, -1]
                    self.oracle.indicator_constraints.add(indvar=iv,
                            lin_expr=[ov, oc], sense='G', rhs=0.0001)

        # class to compile
        if compile_[0].isdigit():
            self.compile = int(compile_)
        else:
            hypos = []
            # determine the true class for the given instance
            for i, (var, val) in enumerate(zip(inputs, datainst)):
                if i in self.free:
                    eql, rhs = [[var], [1]], [val]

                    cnames = ['hypo_{0}'.format(i)]
                    senses = ['E']
                    constr = [eql]

                    assump = self.oracle.linear_constraints.add(lin_expr=constr,
                            senses=senses, rhs=rhs, names=cnames)

                    # adding a constraint to the list of hypotheses
                    hypos.append([cnames[0]])

            self.oracle.solve()
            if self.oracle.solution.is_primal_feasible():
                model = self.oracle.solution

                outvals = [float(model.get_values(o)) for o in self.outputs]
                maxoval = max(zip(outvals, range(len(outvals))))

                # correct class id (corresponds to the maximum computed)
                if compile_ == 'true':
                    self.compile = maxoval[1]
                else:
                    self.compile = int(not maxoval[1])
            else:
                assert 0, 'unsatisfiable instance!'

            # removing the hypotheses
            for hypo in hypos:
                self.oracle.linear_constraints.delete(hypo)

        # linear constraints activating a specific class
        # will be added for each sample individually
        # e.g. self.oracle.linear_constraints.add(lin_expr=[['c_0'], [1]], senses=['G'], rhs=[1])

    def get_counterexample(self, assumptions=[]):
        """
            Extract a (complete) sample corresponding to the given list of
            assumptions.
        """

        assumps = []
        for i, assump in enumerate(assumptions, 1):
            var, val = assump
            eql = [[var], [1]]
            rhs = [int(val)]

            cnames = ['assump_{0}'.format(i)]
            senses = ['E']
            constr = [eql]

            assump = self.oracle.linear_constraints.add(lin_expr=constr,
                    senses=senses, rhs=rhs, names=cnames)

            # adding a constraint to the list of hypotheses
            assumps.append(cnames[0])

        # forcing a wrong class
        self.oracle.linear_constraints.add(lin_expr=[[['c_{0}'.format(self.compile)], [1]]],
                senses='E', rhs=[1], names=['forced_output'])

        # getting the true observation
        # (not the expected one as specified in the dataset)
        self.oracle.solve()
        if self.oracle.solution.is_primal_feasible():
            self.coex_model = self.oracle.solution

            # obtaining input (sample) values
            sample = [round(self.coex_model.get_values(i)) for i in self.inputs]

            outvals = [float(self.coex_model.get_values(o)) for o in self.outputs]
            maxoval = max(zip(outvals, range(len(outvals))))

            # correct class id (corresponds to the maximum computed)
            self.coex_class = maxoval[1]
        else:
            sample = None

        # deleting assumptions
        for assump in assumps:
            self.oracle.linear_constraints.delete(assump)

        # deleting the output enforced
        self.oracle.linear_constraints.delete('forced_output')

        return sample

    def add_sample(self, sample):
        """
            Add constraints for a concrete data sample.
        """

        self.values = sample

        self.hypos = []
        for i, v in enumerate(self.inputs, 1):
            eql = [[v], [1]]
            rhs = [int(sample[i - 1])]

            cnames = ['hypo_{0}'.format(i)]
            senses = ['E']
            constr = [eql]

            if i - 1 in self.free:
                assump = self.oracle.linear_constraints.add(lin_expr=constr,
                        senses=senses, rhs=rhs, names=cnames)

            # adding a constraint to the list of hypotheses
            self.hypos.append(tuple([cnames[0], constr, rhs, senses, i - 1]))

        # observation (forcing it to be wrong)
        self.oracle.linear_constraints.add(lin_expr=[[['c_{0}'.format(self.coex_class)], [1]]],
                senses='E', rhs=[1], names=['forced_output'])

    def explain(self, sample):
        """
            Hypotheses minimization.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # add constraints corresponding to the current sample
        self.add_sample(sample)

        # if satisfiable, then the observation is not implied by the hypotheses
        self.oracle.solve()
        if self.oracle.solution.is_primal_feasible():
            print('  no implication!')
            model = self.oracle.solution
            print('coex  sample:', [self.coex_model.get_values(i) for i in self.inputs])
            print('coex  rounded:', [round(self.coex_model.get_values(i)) for i in self.inputs])
            print('coex classes:', [self.coex_model.get_values(o) for o in self.outputs])
            print('wrong sample:', [model.get_values(i) for i in self.inputs])
            print('wrong rounded:', [round(model.get_values(i)) for i in self.inputs])
            print('wrong classes:', [model.get_values(o) for o in self.outputs])

            sys.exit(1)

        rhypos = self.compute_minimal()

        expl_sz = len(rhypos)
        if self.verbose:
            print('  # hypos left:', expl_sz)

        # removing hypotheses related to the current sample
        for hypo in rhypos:
            self.oracle.linear_constraints.delete(hypo[1])

        # removing the output forced to be wrong
        self.oracle.linear_constraints.delete('forced_output')

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        if self.verbose:
            print('  time: {0:.2f}'.format(self.time))

        return rhypos

    def compute_minimal(self):
        """
            Compute any subset-minimal explanation.
        """

        # result
        rhypos = []

        # simple deletion-based linear search
        for i, hypo in enumerate(self.hypos):
            if i not in self.free:
                continue

            self.oracle.linear_constraints.delete(hypo[0])

            self.oracle.solve()
            if self.oracle.solution.is_primal_feasible():
                # this hypothesis is needed
                # adding it back to the list
                self.oracle.linear_constraints.add(lin_expr=hypo[1],
                        senses=hypo[3], rhs=hypo[2], names=[hypo[0]])

                rhypos.append(tuple([hypo[4], hypo[0]]))

        return rhypos

    def save_image(self, sample, expls, eid, pmap, free, minsz, dataid, compile_):
        """
            Create an image file containing the resulting image with the
            explanations highlighted.
        """

        # image size
        sz = int(math.sqrt(len(sample)))

        # original image
        pixels1, pixels2 = [], []  # this will contain an array of masked pixels
        for i in range(sz):
            row1, row2 = [], []
            for j, v in enumerate(sample[(i * sz):(i + 1) * sz]):
                if v == 1:
                    if i * sz + j in free:
                        row1.append(tuple([0, 255, 255, 230.0]))
                    else:
                        row1.append(tuple([255, 255, 255, 255.0]))

                    row2.append(tuple([255, 255, 255, 255.0]))
                else:
                    if i * sz + j in free:
                        row1.append(tuple([255, 255, 0, 200.0]))
                    else:
                        row1.append(tuple([0, 0, 0, 255.0]))

                    row2.append(tuple([0, 0, 0, 255.0]))

            pixels1.append(row1)
            pixels2.append(row2)

        pixels1 = np.asarray(pixels1, dtype=np.uint8)
        pixels2 = np.asarray(pixels2, dtype=np.uint8)
        mpimg.imsave('sample{0}-patch.png'.format(dataid), pixels1, cmap=mpcm.gray, dpi=5)
        mpimg.imsave('sample{0}-orig.png'.format(dataid), pixels2, cmap=mpcm.gray, dpi=5)

        # most frequent polarity
        polarity = [[0.0, 0.0] for v in range(len(sample))]
        for e in expls:
            for i in e:
                if i[1] == 1:
                    polarity[pmap[i[0]]][1] += 255.0 / len(expls)
                else:
                    polarity[pmap[i[0]]][0] += 255.0 / len(expls)

        pixels1 = []
        if eid != None:
            expl = set([pmap[i[0]] for i in expls[eid]])
            pixels2 = []

        for i in range(sz):
            row1, row2 = [], []
            for j, v in enumerate(sample[(i * sz):(i + 1) * sz]):
                k = i * sz + j

                if polarity[k][0] == polarity[k][1]:
                    if v == 1:
                        row1.append(tuple([255, 255, 255, 255]))
                    else:
                        row1.append(tuple([0, 0, 0, 255]))
                elif polarity[k][0] < polarity[k][1]:
                    row1.append(tuple([255, 0, 0, 255]))
                else:
                    row1.append(tuple([0, 0, 255, 255]))

                if eid != None:
                    if k in free:
                        if k in expl:
                            if polarity[k][0] < polarity[k][1]:
                                row2.append(tuple([255, 0, 0, 255]))
                            else:
                                row2.append(tuple([0, 0, 255, 255]))
                        else:
                                row2.append(tuple([0, 0, 0, 80]))
                    else:
                        if v == 1:
                            row2.append(tuple([255, 255, 255, 255]))
                        else:
                            row2.append(tuple([0, 0, 0, 255]))

            pixels1.append(row1)

            if eid != None:
                pixels2.append(row2)

        pixels1 = np.asarray(pixels1, dtype=np.uint8)
        mpimg.imsave('sample{0}-{1}-all-expls.png'.format(dataid, compile_),
                pixels1, cmap=mpcm.gray, dpi=5)

        if eid != None:
            pixels2 = np.asarray(pixels2, dtype=np.uint8)
            mpimg.imsave('sample{0}-{1}-one-expl.png'.format(dataid, compile_),
                    pixels2, cmap=mpcm.gray, dpi=5)

        if compile_ != 'true':
            expl = set([pmap[i[0]] for i in expls[0]])
            pixels1 = []
            for i in range(sz):
                row1, row2 = [], []
                for j, v in enumerate(sample[(i * sz):(i + 1) * sz]):
                    k = i * sz + j

                    if k in expl:
                        if polarity[k][0] < polarity[k][1]:
                            row1.append(tuple([255, 255, 255, 255]))
                        else:
                            row1.append(tuple([0, 0, 0, 255]))
                    else:
                        if v == 1:
                            row1.append(tuple([255, 255, 255, 255]))
                        else:
                            row1.append(tuple([0, 0, 0, 255]))

                pixels1.append(row1)

            pixels1 = np.asarray(pixels1, dtype=np.uint8)
            mpimg.imsave('sample{0}-true-ae.png'.format(dataid), pixels1,
                    cmap=mpcm.gray, dpi=5)

def parse_options():
    """
        Parses command-line options:
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'c:d:e:i:hp:v',
                                   ['compile=',
                                    'data=',
                                    'expl=',
                                    'intervals=',
                                    'htype=',
                                    'patch=',
                                    'save',
                                    'help',
                                    'verb'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    compile_ = 0
    datainst = 0
    expl = None
    free = None
    intervals = 8
    save = False
    htype = 'lbx'
    verb = 0

    for opt, arg in opts:
        if opt in ('-c', '--compile'):
            compile_ = str(arg)
        elif opt in ('-d', '--data'):
            datainst = int(arg)
        elif opt in ('-e', '--expl'):
            expl = int(arg) if arg != 'none' else None
        elif opt in ('-i', '--intervals'):
            intervals = int(arg)
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif opt == ('--htype'):
            htype = str(arg)
        elif opt in ('-p', '--patch'):
            free = str(arg)
        elif opt == ('--save'):
            save = True
        elif opt in ('-v', '--verb'):
            verb += 1
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return compile_, datainst, expl, free, intervals, htype, save, verb, args


def usage():
    """
        Prints usage message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] lp-file')
    print('Options:')
    print('        -c, --compile=<int>        Class to compile')
    print('                                   Available values: [0 .. INT_MAX], true, opposite (default: 0)')
    print('        -d, --data=<int>           Index of data instance to work with')
    print('                                   Available values: [0 .. INT_MAX] (default: 0)')
    print('        -e, --expl=<int>           Show explanation with this numerical identifier')
    print('                                   Available values: [0 .. INT_MAX], none (default: none)')
    print('        -i, --intervals=<int>      Number of intervals, i.e. values, per pixel')
    print('                                   Available values: [2 .. INT_MAX] (default: 8)')
    print('        -h, --help')
    print('        --htype=<string>           Approach to enumerate hitting sets')
    print('                                   Available values: lbx, maxsat/rc2/sorted, mcsls (default: lbx)')
    print('        -p, --patch=<string>       A path to a file containing a comma-separated list of free pixels identifiers')
    print('                                   Default: none (means that all pixels are free)')
    print('        --save                     Save the resulting image')
    print('        -v, --verb                 Be verbose')


if __name__ == '__main__':
    # parse command-line options
    compile_, dataid, eid, free, intervals, htype, save, verb, files = parse_options()

    if files:
        if free:  # let's get the ids of free pixels
            with open(free, 'r') as fp:
                free = [int(p.strip()) for p in ''.join([l.strip() for l in fp.readlines()]).split(',')]

        ilp, inputs, outputs, data = parse_lpfile(files[0])

        # feature names
        names = data[0].split(',')

        datainst = [int(float(v)) for v in data[dataid + 1].split(',')]

        if not free:
            free = list(range(len(inputs)))

        pmap = {inputs[p]: p for p in free}

        # hitting set enumerator
        h = prepare_hitman(free, inputs, intervals, htype)

        # ILP-based explainer
        x = ILPExplainer(ilp, inputs, outputs, names, datainst, free,
                compile_, verb)

        expls, coexs = compile_classifier(h, x, inputs, pmap, intervals,
                verb)

        if expls and coexs:
            print('')
            print('# pfree:', len(free))
            print('# expls:', len(expls))
            print('# coexs:', len(coexs))
            print('')
            print('min expl:', len(min(expls, key=lambda x: len(x))))
            print('max expl:', len(max(expls, key=lambda x: len(x))))

            elens = list(map(lambda x: len(x), expls))
            print('avg expl: {0:.2f}'.format(float(sum(elens)) / len(elens)))

            print('')
            print('compiled class:', x.compile, '({0})'.format(compile_))

            if save:
                x.save_image(datainst, expls, eid, pmap, set(free), min(elens), dataid, compile_)
