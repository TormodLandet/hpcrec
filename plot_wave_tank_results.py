import numpy


def plot_free_surface(xpos, eta, tvec, title):
    from matplotlib import pyplot
    from matplotlib.widgets import Slider
    
    fig, ax = pyplot.subplots()
    pyplot.subplots_adjust(bottom=0.25)
    axcolor = 'lightgoldenrodyellow'
    slider_ax = pyplot.axes([0.1, 0.1, 0.8, 0.03], axisbg=axcolor)
    slider = Slider(slider_ax, 'Time', tvec[0], tvec[-1], valinit=tvec[0])
    
    xmin = xpos[0]
    xmax = xpos[-1]
    ymin = eta.min()
    ymax = eta.max()
    xdiff = xmax - xmin 
    ydiff = ymax - ymin
    xmin, xmax = xmin - 0.1*xdiff, xmax + 0.1*xdiff
    ymin, ymax = ymin - 0.1*ydiff, ymax + 0.1*ydiff
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    
    line, = ax.plot(xpos, eta[0])
    
    def update(val):
        #xmin, xmax = ax.get_xlim()
        #ymin, ymax = ax.get_ylim()
        
        t = slider.val
        it = numpy.argmin(abs(tvec - t))
        line.set_ydata(eta[it])
        
        #ax.set_xlim(xmin, xmax)
        #ax.set_ylim(ymin, ymax)
        #ax.legend(loc='lower right')
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    slider.set_val(tvec[0])


def read_wave_tank__out(file_name):
    with open(file_name, 'rt') as inp:
        inp.readline()
        xpos = numpy.array([float(v) for v in inp.readline().split()])
        inp.readline()
        tvec = numpy.array([float(v) for v in inp.readline().split()])
        wds = inp.readline().split()
        N = int(wds[1])
        M = int(wds[2])
        eta = numpy.zeros((N, M), float)
        for i in range(N):
            wds = inp.readline().split()
            for j in range(M):
                eta[i,j] = float(wds[j])
    
    return xpos, tvec, eta


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_name = sys.argv
    else:
        file_name = 'result_wave_tank_demo.out'
    
    xpos, tvec, eta = read_wave_tank__out(file_name)
    plot_free_surface(xpos, eta, tvec, 'Free surface elevation')
    
    from matplotlib import pyplot
    pyplot.show()
    