import numpy
from matplotlib import pyplot


def ts_zero_upcrossing_period(t, I):
    duration = t[I[-1]] - t[I[0]]
    return duration/(len(I)-1)


def ts_peaks(ts, I):
    N = len(I)
    J = numpy.zeros(N-1, int)
    for i in range(N-1):
        J[i] = I[i] + numpy.argmax(ts[I[i]:I[i+1]])
    return J


def ts_zero_upcrossings(ts):
    I = []
    for i in range(1, len(ts)):
        if ts[i-1] < 0 and ts[i] >= 0:
            I.append(i)
    return numpy.array(I, int)


class LogfileReader(object):
    def __init__(self, filename):
        timeseries = {'t': [], 'Fp0': [], 'Fp1': [], 'Fv0': [], 'Fv1': []}
        inp = {}
        
        # Read data from logfile
        for line in open(filename, 'rt'):
            if line.startswith('    ') and line.count('=') == 1:
                wds = line.split()
                name = wds[0]
                K = 4 + len(name) + 3
                value_str = line[K:].strip()
                value = eval(value_str)
                inp[name] = value
                continue
            elif 'Fv' not in line or line[-1] != '\n':
                continue
            wds = line.split()
            ip = wds.index('Fp:')
            iv = wds.index('Fv:')
            
            timeseries['t'].append(float(wds[3]))
            timeseries['Fp0'].append(float(wds[ip+1]))
            timeseries['Fp1'].append(float(wds[ip+2]))
            timeseries['Fv0'].append(float(wds[iv+1]))
            timeseries['Fv1'].append(float(wds[iv+2]))
            
        # Make arrays
        for name in 't Fp0 Fp1 Fv0 Fv1'.split():
            timeseries[name] = numpy.array(timeseries[name], float)
        
        self.input = inp
        self.timeseries = timeseries


def plot_logfile(inpfile, tstart):
    # Read the log
    log = LogfileReader(inpfile)
    inp = log.input
    timeseries = log.timeseries
    
    # Get parameters
    d = inp.get('d', 0.1)
    U0 = inp.get('U0', 0.1)
    rho = inp.get('rho', 1)
    #td1, _td2, td3 = inp.get('disturbance_time', (0, 0, 0))
    td1 = td3 = 0
    
    # Filter the time series
    t = timeseries['t']
    sieve = (t > tstart) & ((t < td1) | (t > td3))
    t =  t[sieve]
    
    fig = pyplot.figure()
    axes = [fig.add_subplot(n) for n in (211, 212)]
    legends = [[], []]
    
    print 'Input:'
    for name, value in sorted(inp.items()):
        print '    %s = %r' % (name, value)
    
    for name in 'Fp0 Fp1 Fv0 Fv1'.split():
        data = timeseries[name][sieve]
        mean = data.mean()
        data2 = data - mean
        I = ts_zero_upcrossings(data2)
        scale = 2/(rho*U0**2*d)
        
        direction = int(name[-1])
        pyplot.sca(axes[direction])
        
        data_plot = data2
        line, = pyplot.plot(t, data_plot*scale)
        legends[direction].append((line, name))
        pyplot.title('Force in %s direction' % ('lift' if direction else 'drag'))
        
        if len(I) > 2:
            J = ts_peaks(data2, I)
            peaks = data[J]
            Tz = ts_zero_upcrossing_period(t, I)
            a = peaks.mean()    
            print name
            print '    mean', mean*scale
            print '    Tz  ', Tz
            print '    ampl', a*scale
            print '      St', d/(Tz*U0)
            print '     rms', ((data*scale)**2).mean()**0.5    
            pyplot.plot(t[I], data_plot[I]*scale, 'bo')
            pyplot.plot(t[J], data_plot[J]*scale, 'ro')
    
    for ax, leg in zip(axes, legends):
        pyplot.sca(ax)
        lines, names = zip(*leg)
        pyplot.legend(lines, names)
    pyplot.tight_layout()
    
    print 'Tmax', t[-1]
    pyplot.show()


def plot_multiple(args, tstart):
    name = args[-1]
    assert name in 'Fp0 Fp1 Fv0 Fv1'.split()
    
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    legends = []
    
    for inpfile in args[:-1]:
        # Read the log
        log = LogfileReader(inpfile)
        inp = log.input
        timeseries = log.timeseries
    
        # Get parameters
        d = inp.get('d', 0.1)
        U0 = inp.get('U0', 0.1)
        rho = inp.get('rho', 1)
        #td1, _td2, td3 = inp.get('disturbance_time', (0, 0, 0))
        td1 = td3 = 0
    
        # Filter the time series
        t = timeseries['t']
        sieve = (t > tstart) & ((t < td1) | (t > td3))
        t =  t[sieve]
        
        data = timeseries[name][sieve]
        mean = data.mean()
        data2 = data - mean
        I = ts_zero_upcrossings(data2)
        scale = 2/(rho*U0**2*d)
        
        direction = int(name[-1])
        pyplot.sca(ax)
        
        data_plot = data
        line, = pyplot.plot(t, data_plot*scale)
        legends.append((line, inpfile))
        pyplot.title('Force in %s direction (%s)' % ('lift' if direction else 'drag', name))
        
        if len(I) > 2:
            J = ts_peaks(data2, I)
            peaks = data[J]
            Tz = ts_zero_upcrossing_period(t, I)
            a = peaks.mean()    
            print inpfile, name
            print '    mean', mean*scale
            print '    Tz  ', Tz
            print '    ampl', a*scale
            print '      St', d/(Tz*U0)
            print '     rms', ((data*scale)**2).mean()**0.5    
            #pyplot.plot(t[I], data_plot[I]*scale, 'bo')
            #pyplot.plot(t[J], data_plot[J]*scale, 'ro')
            print 'Tmax', t[-1]
            print
    
    lines, names = zip(*legends)
    pyplot.legend(lines, names)
    pyplot.tight_layout()
    
    pyplot.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logfiles', default=['cylinder.log'], nargs='*')
    parser.add_argument('--tstart', type=float, default=0, help='steady state start')
    args = parser.parse_args()
    
    
    pyplot.style.use('ggplot')
    tstart = args.tstart
    inpfiles = args.logfiles
    
    if len(inpfiles) == 1:
        plot_logfile(inpfiles[0], tstart)
    else:
        plot_multiple(inpfiles, tstart)
        
