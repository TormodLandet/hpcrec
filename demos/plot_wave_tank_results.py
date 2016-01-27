from __future__ import division
import cPickle as pickle
from math import sinh, cosh, tanh, cos
import numpy
from matplotlib import pyplot
from matplotlib.widgets import Slider
from wave_tank_linear import WaveTankInput # for pickle


def plot_free_surface(wti, xpos, eta, tvec):
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
    ax.set_title('Free surface elevation')
    
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
        
    # Linear wave maker theory results for comparison
    xpos = numpy.array(xpos)
    analytical = numpy.zeros_like(xpos)
    g = wti.g
    h = wti.h
    for ia, s in enumerate(wave_tank_input.wm_ampls):  
        w = wave_tank_input.wm_freqs[ia]
        #b = wave_tank_input.wm_phases[ia]
        k = wave_number(w, g, h)
        kh = k*h
        a = 4*sinh(kh)/kh*(kh*sinh(kh) - cosh(kh) + 1)/(sinh(2*kh) + 2*kh)*s
        analytical[:] += a*numpy.sin(k*xpos)
    ax.plot(xpos, analytical)
    
    slider.on_changed(update)
    slider.set_val(tvec[0])


def plot_wave_maker(wti, xpos, eta, tvec):
    wave_maker_speed = numpy.zeros_like(tvec)
    Nt = tvec.size
    for it in range(1, Nt):
        t = tvec[it]
        
        # Calculate wave maker amplitude at y = h (still water height)
        ramp = min(t/wti.tramp, 1)
        for ia, wm_ampl in enumerate(wave_tank_input.wm_ampls):  
            wm_freq = wave_tank_input.wm_freqs[ia]
            wm_phase = wave_tank_input.wm_phases[ia]
            wave_maker_speed[it] += wm_freq*wm_ampl*cos(wm_freq*t + wm_phase)*ramp
    
    pyplot.figure()
    pyplot.plot(tvec, wave_maker_speed)


def wave_number(freq, g, h, tol=1e-8):
    "Calculate the wave number k by Picard iteration"
    k0 = freq**2/g
    k = k0
    err = 1
    while err > tol:
        k2 = k0/tanh(k*h)
        err = abs(k-k2)
        k = k2
    return k2


def read_wave_tank_out(file_name):
    with open(file_name, 'rb') as inp:
        data = pickle.load(inp)
    return data['wave_tank_input'], data['xpos'], data['tvec'], data['eta']


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = 'result_wave_tank_demo.out'
    
    wave_tank_input, xpos, tvec, eta = read_wave_tank_out(file_name)
    plot_free_surface(wave_tank_input, xpos, eta, tvec)
    #plot_wave_maker(wave_tank_input, xpos, eta, tvec)
    pyplot.show()
