""" A not so simple topcat: interactive selection between plots """
#============================ LASSO PLOT ==============================
from matplotlib.widgets import Lasso
from matplotlib.nxutils import points_inside_poly
from numpy import nonzero, array
import pylab as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from matplotlib.cbook import silent_list
from eztables import Table


#=================================================================================
class Sample(object):
    """ Handle a sample of data with its associated representations in 1d and 2d """
#=================================================================================
    def __init__(self, data, idx=None, f1d=None, f2d=None, name=None):
        self.data = data
        self.name = name or 'sample'
        if idx is not None:
            self.idx = idx
        else:
            self.idx = range(data.nrows)

        if f1d is not None:
            self.set_f1d(f1d[0], **f1d[1])
        else:
            self.set_f1d(plt.hist, {'label': self.name})

        if f2d is not None:
            self.set_f2d(f2d[0], **f2d[1])
        else:
            self.set_f2d(plt.plt, {'label': self.name})

    def set_name(self, name):
        self.name = name

    def set_f2d(self, func=None, **kwargs):
        if 'label' not in kwargs:
            kwargs['label'] = self.name
        if 'picker' not in kwargs:
            kwargs['picker'] = 5
        if func is not None:
            if not hasattr(func, '__call__'):
                raise TypeError('Expecting a callable')
        self.f2d = (func, kwargs)

    def set_f1d(self, func=None, **kwargs):
        if 'label' not in kwargs:
            kwargs['label'] = self.name
        if 'picker' not in kwargs:
            kwargs['picker'] = 5
        if func is not None:
            if not hasattr(func, '__call__'):
                raise TypeError('Expecting a callable')
        self.f1d = (func, kwargs)

    def __len__(self):
        return np.size(np.ravel(self.idx))

    def __repr__(self):
        return 'Sample {}: {} points\n \t 1d visual: {}, {}\n \t 2d visual: {}, {}'.format(self.name, len(self), self.f1d[0].__name__, self.f1d[1], self.f2d[0].__name__, self.f2d[1])

    def draw_1d(self, x, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        _f = getattr(ax, self.f1d[0].__name__, None)
        if _f is not None:
            return _f(x[self.idx], **(self.f1d[1]))
        else:
            return self.f1d[0](x[self.idx], ax=ax, **(self.f1d[1]))

    def draw_2d(self, x, y, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        _f = getattr(ax, self.f2d[0].__name__, None)
        if _f is not None:
            return _f(x[self.idx], y[self.idx], **(self.f2d[1]))
        else:
            return self.f2d[0](x[self.idx], y[self.idx], ax=ax, **(self.f2d[1]))

    def draw(self, x, y=None, ax=None, **kwargs):
        if y is not None:
            return self.draw_2d(x, y, ax=ax, **kwargs)
        else:
            return self.draw_1d(x, ax=ax, **kwargs)


#=================================================================================
class View(object):
    """ Class to handle one axes instance with interactivity and selections """
#=================================================================================
    def __init__(self, data, xt, yt=None, ax=None,
                 f1d=plt.hist, f2d=plt.plot, f1d_kwargs={}, f2d_kwargs={},
                 idx=None, name=None, selections=None, parent=None):

        self.axes = ax or plt.gca()

        if not isinstance(self.axes, Axes):
            raise ValueError('type {} not managed, only matplotlib.axes.Axes are handled'.format(type(ax)))

        #get the data into a Table format for fast interactions
        if not isinstance(data, Table):
            self.data = Table(data)
        else:
            self.data = data

        #define axes and precompute the data
        self.xt = xt
        self.yt = yt
        self.x = self.data.evalexpr(self.xt)
        if yt is not None:
            self.y = self.data.evalexpr(self.yt)
        else:
            self.y = None

        #set the default values
        self.name = name                       # or ax.name
        self.canvas = self.axes.figure.canvas  # keep the current canvas for interactions
        self.fig = self.axes.figure            # keep the current figure ling
        self.selections = selections or {}     # indices of the selections
        self._draws = {}                       # keep track of existing representations
        self._next_select = []                 # set when using lasso interaction to define the plotting
        self.parent = parent                   # used when within a DataFrame
        self._lasso = False                    # set during lasso interaction

        #update canvas
        self.xlabel(xt.replace('_', ' '))
        if yt is not None:
            self.ylabel(yt.replace('_', ' '))

        #add initial plot
        if len(self.selections) == 0:
            idx = idx or range(self.data.nrows)
            self.add_sample(idx, f1d, f2d, f1d_kwargs, f2d_kwargs)
        else:
            self.pchanged()

        #connect to callbacks
        self.canvas.mpl_connect('pick_event', self.onpick)
        self.canvas.mpl_connect('button_press_event', self.onpress)
        self.canvas.mpl_connect('button_release_event', self.onrelease)

    def xlabel(self, txt):
        """ Set xlabel"""
        if txt is not None:
            self.axes.set_xlabel(txt)

    def ylabel(self, txt):
        """ set ylabel """
        if txt is not None:
            self.axes.set_ylabel(txt)

    def onpick(self, event):
        """ Define callback Onpick """
        if self._next_select > 0:
            return

        if isinstance(event.artist, Rectangle):
            patch = event.artist
            print('onpick patch:', patch.get_path())
        else:
            ind = event.ind
            print 'onpick data:', ind   # , np.take(x, ind), np.take(y, ind)

        self.pick = event

    def add_sample(self, idx, f1d=plt.hist, f2d=plt.plot, f1d_kwargs={}, f2d_kwargs={}, replace=True, **kwargs):
        """ add a selection to the view, selection name is given by label kewyword or a default one it given"""

        if 'label' in kwargs:
            lbl = kwargs['label']
        else:
            lbl = 's%d' % (len(self.selections) + 1)

        if (replace is True) | (lbl not in self.selections):
            self.selections[lbl] = Sample(self.data, idx=idx, f1d=(f1d, f1d_kwargs), f2d=(f2d, f2d_kwargs), name=lbl)
        else:
            raise KeyError('Sample {} already registered.'.format(lbl))

        return self.draw_selection(lbl, replace=replace)

    def draw_selection(self, lbl, replace=True):
        """ draw a given subset """
        if (lbl not in self._draws) | (replace is True):
            sample = self.selections[lbl]

            self._draws[lbl] = sample.draw(self.x, self.y, ax=self.axes)

            if plt.isinteractive():
                self.draw()

            return sample, self._draws[lbl]

    def select(self, f1d=plt.hist, f2d=plt.plot, f1d_kwargs={}, f2d_kwargs={}):
        """ Start a selection """
        self._next_select = [f1d, f2d, f1d_kwargs, f2d_kwargs]

    def where(self, condition, condvars=None, start=None, stop=None, step=None, f1d=plt.hist, f2d=plt.plot, f1d_kwargs={}, f2d_kwargs={}, *args, **kwargs):
        ind = self.data.where(condition, condvars=condvars, start=start, stop=stop, step=step, *args)
        return self.add_sample(ind, f1d=f1d, f2d=f2d, f1d_kwargs=f1d_kwargs, f2d_kwargs=f2d_kwargs, **kwargs)

    def __lasso_callback__(self, verts):
        """ Callback for the lasso """
        if self.y is not None:
            xys = array([self.x, self.y]).T
            mask = points_inside_poly(xys, verts)
            ind = nonzero(mask)[0]
        else:
            _verts = np.asarray(verts)
            xmin = _verts[:, 0].min()
            xmax = _verts[:, 0].max()
            ind = np.where((self.x >= xmin) & (self.x <= xmax))
        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self._lasso)
        del self._lasso
        self.ind = ind
        (f1d, f2d, f1d_kwargs, f2d_kwargs) = self._next_select
        self.add_sample(ind, f1d=f1d, f2d=f2d, f1d_kwargs=f1d_kwargs, f2d_kwargs=f2d_kwargs)
        while len(self._next_select) > 0:
            del self._next_select[0]
        self.draw()
        self.pchanged()

    def onpress(self, event):
        """ on press callback, will run lasso """
        if event.inaxes != self.axes:
            return
        if len(self._next_select) > 0:
            if self.canvas.widgetlock.locked():
                return
            self._lasso = Lasso(event.inaxes, (event.xdata, event.ydata), self.__lasso_callback__)
            # acquire a lock on the widget drawing
            self.canvas.widgetlock(self._lasso)
            self.pchanged()

    def onrelease(self, evt):
        """canvas mouseup handler
        """
        print 'Released'
        if self.parent is not None:
            self.parent.pchanged()
        self.canvas.draw()

    def show(self):
        plt.show()

    def draw(self):
        self.canvas.draw()

    def set_visible(self, lbl, val):
        """ make a subset visible or not """
        if lbl in self._draws:
            for k in self._draws[lbl]:
                if type(k) not in [np.ndarray]:
                    if isinstance(k, silent_list):
                        for j in k:
                            j.set_visible(val)
                    else:
                        k.set_visible(val)
        self.draw()

    def clear(self):
        """ Clear a view and related data """
        self._draws.clear()
        self.selections.clear()
        self.axes.clear()

    def del_selection(self, lbl):
        """ remove a subset """
        self.set_visible(lbl, False)
        if lbl in self._draws:
            self._draws.pop(lbl)
        if lbl in self.selections:
            self.selections.pop(lbl)

    def pchanged(self, fromparent=False):
        """ propagate changes """
        if (fromparent is True) | (self.parent is None):
            for k in self.selections:
                self.draw_selection(k, replace=False)
            self.draw()
        elif (self.parent is not None) | (fromparent is False):
            self.parent.pchanged()


#=================================================================================
class DataFrame(object):
    """ handle plots with linked data """
#=================================================================================
    def __init__(self, data):
        self.data = Table(data)
        self.views = {}
        self.selections = {}
        self._next_select = []

    def get_view(self, name, **kwargs):
        """ return a given view from its name or idx """
        if type(name) == str:
            return self.views[name]
        if type(name) == int:
            return self.views.values()[name]
        elif isinstance(name, Axes):
            if name.name in self.axes:
                return self.axes[name.name]
            else:
                return self.add_view(ax=name, **kwargs)
        else:
            return

    def add_view(self, view, name=None, **kwargs):
        """ Add a new managed view """
        #default name
        if not isinstance(view, View):
            raise ValueError('type {} not managed'.format(type(name)))

        if (name is None):
            name = 'view_{}'.format(len(self.views) + 1)
        view.name = name

        #check if already registered
        if self.existing_view(name):
            raise KeyError('name {} already exists'.format(name))

        #add parent definition
        view.parent = self

        self.views[name] = view
        if len(self.selections) == 0:
            self.selections = view.selections
        else:
            view.selections = self.selections
            view._next_select = self._next_select

        self.pchanged()

        return view

    def add_sample(self, idx, f1d=plt.hist, f2d=plt.plot, f1d_kwargs={}, f2d_kwargs={}, replace=True, **kwargs):
        """ add a selection to the view, selection name is given by label kewyword or a default one it given"""

        if 'label' in kwargs:
            lbl = kwargs['label']
        else:
            lbl = 's%d' % (len(self.selections) + 1)

        if (replace is True) | (lbl not in self.selections):
            self.selections[lbl] = Sample(self.data, idx=idx, f1d=(f1d, f1d_kwargs), f2d=(f2d, f2d_kwargs), name=lbl)
        else:
            raise KeyError('Sample {} already registered.'.format(lbl))

        return self.pchanged()

    def where(self, condition, condvars=None, start=None, stop=None, step=None, f1d=plt.hist, f2d=plt.plot, f1d_kwargs={}, f2d_kwargs={}, *args, **kwargs):
        ind = self.data.where(condition, condvars=condvars, start=start, stop=stop, step=step, *args)
        return self.add_sample(ind, f1d=f1d, f2d=f2d, f1d_kwargs=f1d_kwargs, f2d_kwargs=f2d_kwargs, **kwargs)

    def select(self, f1d=plt.hist, f2d=plt.plot, f1d_kwargs={}, f2d_kwargs={}):
        """ Start a selection """
        self._next_select = [f1d, f2d, f1d_kwargs, f2d_kwargs]
        for vk in self.views.values():
            vk._next_select = self._next_select

    def subplot(self, xt, yt=None, subplot=111, f1d=plt.hist, f2d=plt.plot, f1d_kwargs={}, f2d_kwargs={}, idx=None, fig=None, ax=None, name=None, **kwargs):
        """ Create a subplot command, creating axes with::

        subplot(numRows, numCols, plotNum)

        where *plotNum* = 1 is the first plot number and increasing *plotNums*
        fill rows first.  max(*plotNum*) == *numRows* * *numCols*

        You can leave out the commas if *numRows* <= *numCols* <=
        *plotNum* < 10, as in::

        subplot(211)    # 2 rows, 1 column, first (upper) plot

        ``subplot(111)`` is the default axis.
        """
        fig = fig or plt.gcf()
        ax = fig.add_subplot(subplot, **kwargs)
        view = View(self.data, xt, yt=yt, idx=idx, ax=ax, name=name,
                    f1d=f1d, f2d=f2d, f1d_kwargs=f1d_kwargs, f2d_kwargs=f2d_kwargs,
                    selections=self.selections, parent=self)
        return self.add_view(view, name=view.name)

    def figure(self, xt, yt=None, f1d=plt.hist, f2d=plt.plot, f1d_kwargs={}, f2d_kwargs={}, idx=None, name=None, **kwargs):
        """ Create a new figure and return a :class:`matplotlib.figure.Figure` instance.
        """
        fig = plt.figure(**kwargs)
        return self.subplot(xt, yt, subplot=111,
                           f1d=f1d, f2d=f2d, f1d_kwargs=f1d_kwargs, f2d_kwargs=f2d_kwargs,
                           idx=idx, fig=fig, name=name, **kwargs)

    def pop_view(self, name):
        """ unregister a  view """
        return self.views.pop(name)

    def existing_view(self, name):
        """ check existing view """
        if type(name) == str:
            return name in self.views
        elif isinstance(name, View):
            return name.name in self.views
        elif isinstance(name, int):
            return name < len(self.views)
        else:
            raise ValueError('type {} not managed'.format(type(name)))

    def pchanged(self, skip=[]):
        """ propagate changes """
        for vn, vk in self.views.iteritems():
                vk.pchanged(fromparent=True)

    def del_selection(self, lbl):
        """ remove a subset """
        for vn, vk in self.views.iteritems():
            vk.del_selection(lbl)

    def set_visible(self, lbl, val):
        for vn, vk in self.views.iteritems():
            vk.set_visible(lbl, val)

if __name__ == '__main__':
    # create a framework
    df = DataFrame('./temp.cont.zsun.fits')
    # add some 2d plots directly using column names
    # and even some operations such as
    # log10(AGE) and differences
    a = df.subplot('log10(AGE)', 'HST_WFPC2_F555W', subplot=221)
    b = df.subplot('HST_WFPC2_F336W-HST_WFPC2_F555W', 'HST_WFPC2_F336W', subplot=222)
    # add an histogram as well
    c = df.subplot('log10(AGE)', subplot=223)

    # add one selection from a simple where call
    ## see how it propagates to all the plots
    df.where('AGE >= 1000', f2d_kwargs=dict(lw=2., marker='o'))

    # manually select a sample by drawing on one subplot
    df.select(
        f2d=plt.plot, f2d_kwargs=dict(alpha=1., color='red', marker='o'),
        f1d=plt.hist, f1d_kwargs=dict(color='red', alpha=0.5)
    )
