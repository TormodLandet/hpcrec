import sys
import argparse
import numpy
from matplotlib import pyplot

parser = argparse.ArgumentParser()
parser.add_argument('logfile', default='cylinder.log')
parser.add_argument('--tstart', type=float, default=0, help='steady state start')
args = parser.parse_args()


pyplot.style.use('ggplot')
tstart = args.tstart
inpfile = args.logfile


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


def main():
    global inpfile
    if sys.argv[1:]:
        inpfile = sys.argv[1]
    
    timeseries = {'t': [], 'Fp0': [], 'Fp1': [], 'Fv0': [], 'Fv1': []}
    inp = {}
    for line in open(inpfile, 'rt'):
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
    
    d = inp.get('d', 0.1)
    U0 = inp.get('U0', 0.1)
    rho = inp.get('rho', 1)
    td1, _td2, td3 = inp.get('disturbance_time', (0, 0, 0))
    
    for name in 't Fp0 Fp1 Fv0 Fv1'.split():
        timeseries[name] = numpy.array(timeseries[name], float)
    t = timeseries['t']
    sieve = (t > tstart) & ((t < td1) | (t > td3))
    t =  t[sieve]
    
    figures = [pyplot.figure(), pyplot.figure()]
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
        fig = figures[direction]
        pyplot.sca(fig.gca())
        
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
    
    for fig, leg in zip(figures, legends):
        pyplot.sca(fig.gca())
        lines, names = zip(*leg)
        pyplot.legend(lines, names)
    
    print 'Tmax', t[-1]
    pyplot.show()


if __name__ == '__main__':
    main()
