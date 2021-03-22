import os
import sys
import random
import numpy as np

from collections import OrderedDict
from tabulate import tabulate
from pandas import DataFrame
from time import gmtime, strftime
import pandas as pd

letters = "abcdefghijklmnopqrstuvwxyz"

class Logger:
    def __init__(self, name='name', fmt=None, base='./logs', path=None):
        self.handler = True
        self.scalar_metrics = OrderedDict()
        self.fmt = fmt if fmt else dict()

        time, fname = gmtime(), '-'.join(sys.argv[0].split('/')[-1:])
        if path is not None:
            self.path = path
        else:
            self.path = '%s/%s-%s-%s-%s' % (base, fname, name, strftime('%m-%d-%H:%M:%S', time), "".join([random.choice(letters) for i in range(5)]))

        if not os.path.exists(base):
            os.makedirs(base)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.logs = self.path + '/log.csv'
        self.output = self.path + '/output.out'
        self.checkpoint = self.path + '/model.cpt'

        def prin(*args):
            str_to_write = ' '.join(map(str, args))
            with open(self.output, 'a') as f:
                f.write(str_to_write + '\n')
                f.flush()

            print(str_to_write)
            sys.stdout.flush()

        self.print = prin
        
    def get_checkpoint(self, epoch):
        return self.path + '/model_ep%d.cpt' % epoch

    def add_scalar(self, t, key, value):
        if key not in self.scalar_metrics:
            self.scalar_metrics[key] = []
        self.scalar_metrics[key] += [(t, value)]

    def add_dict(self, t, d):
        for key, value in d.iteritems():
            self.add_scalar(t, key, value)

    def add(self, t, **args):
        for key, value in args.items():
            self.add_scalar(t, key, value)

    def iter_info(self, order=None):
        names = list(self.scalar_metrics.keys())
        if order:
            names = order
        values = [self.scalar_metrics[name][-1][1] for name in names]
        t = int(np.max([self.scalar_metrics[name][-1][0] for name in names])) 
        fmt = ['%s'] + [self.fmt[name] if name in self.fmt else '.7f' for name in names]

        if self.handler:
            self.handler = False
            self.print(tabulate([[t] + values], ['epoch'] + names, floatfmt=fmt))
        else:
            self.print(tabulate([[t] + values], ['epoch'] + names, tablefmt='plain', floatfmt=fmt).split('\n')[1])

    def save(self, silent=False):
        result = None
        for key in self.scalar_metrics.keys():
            if result is None:
                result = DataFrame(self.scalar_metrics[key], columns=['t', key]).set_index('t')
            else:
                df = DataFrame(self.scalar_metrics[key], columns=['t', key]).set_index('t')
                result = pd.concat([result, df], axis=1)
        result.to_csv(self.logs)
        if not silent:
            self.print('The log/output/model have been saved to: ' + self.path + ' + .csv/.out/.cpt')
