import numpy
import scipy

def get_pulse_average_and_rms(tau, FWHM, HannApo):
    hann = lambda t: 0.5*(1-numpy.cos(2*numpy.pi*t/tau))
    
    sigma = numpy.sqrt(2*numpy.log(2) / (numpy.pi*FWHM)**2)
    gauss = lambda t: numpy.exp(-(t-tau/2)**2 / (2*sigma**2))
    
    if HannApo:
        if FWHM == 0: # pure Hann
            satPulse = hann
        else: # Gauss-Hann
            satPulse = lambda t: gauss(t) * hann(t)
    else: # pure Gauss
        satPulse = gauss
    
    average = scipy.integrate.quad(satPulse, 0, tau)[0]/tau
    rms = numpy.sqrt(
        scipy.integrate.quad(lambda t: satPulse(t)**2, 0, tau)[0]/tau)

    return average, rms
