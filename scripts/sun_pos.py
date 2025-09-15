# -*- coding: utf-8 -*-

print('here is sun_pos')

import numpy as np
from math import atan2
import datetime
import matplotlib.pyplot as plt
import pylab
from pylab import *
from numba import jit

# -*- coding: utf-8 -*-


def  solarposnd(latitude,longitude,day,time,year,timezone,limb):
    """
    Fonction qui calcule la position du soleil 

    Id: solarposnd.pro,v 5.4
    
    Copyright (c) 2001 Javier G Corripio.
    jgc@geo.ed.ac.uk
    +
    NAME:
        SolarPosnd
    
    PURPOSE:
        computes solar vector, azimuth and elevation for a given day, time,
        latitude and longitude
    
    CATEGORY:
        Solar Radiation
    
    CALLING SEQUENCE:
        Solpos = SolarPos(latitude,longitude,day,time,year,timezone,limb)
    
    INPUTS:
        latitude: (degrees, 90 : -90) FLOAT
        longitude: (degrees, -180 : 180) FLOAT
        day: day of the year (1 : 365)
          = present julian day (at noon)-Julian day(1/1/present year)  FLOAT
        time: (0.0 : 23.99) FLOAT
        time zone: Greenwich=0 INT
        year
        LIMB: set this keyword to compute sunset and sunrise for the upper limb
            of the sun, including refraction (+/-(16'+34') )
    
    OUTPUTS:
        data structure SOLPOS:
            solpos{1,2}vector  DOUBLE[3]
            solpos{2,2}rho2 reciprocal of the square of the radius
                    vector of the earth
            solpos{3,2}azimuth
            solpos{4,2}zenith
            solpos{5,2}declination
            solpos{6,2}sunrise
            solpos{7,2}sunset
            solpos{8,2}EqTime
            solpos{9,2}omega
    
    
    Bourgues, B.: 1985, Improvement in solar declination computation,
    Solar Energy 35(4), 367 369.
    
    Corripio, J. G.: 2001, Vectorial algebra algorithms for calculating terrain
    parameters from DEMs and the position of the sun for solar radiation
    modelling in mountainous terrain, International Journal of Geographical
    Information Science . Submitted.
    
    Iqbal, M.: 1983, An Introduction to Solar Radiation, Academic Press, Toronto.
    
    Spencer, J. W.: 1971, Fourier series representation of the position of
    the sun, Search 2, 172.
    
    U.S. Naval Observatory Astronomical Applications Department
    http://aa.usno.navy.mil/faq/docs/RST_defs.html
    
    
    EXAMPLE:
        solpos=SolarPos(55,-3,311,12.0,/limb)
            solar position at EH 7/11 noon, will give sunrise and set
            including refraction and ocultation of the upper limb
    
    MODIFICATION HISTORY:
        Created Javier Corripio, August, 2001
        modified 11/
    Marie Dumont 25/01/08
    """

    radeg = np.pi/180.
    # sun-earth distance, eccentricity correction (Spencer 1971)
    dayang_S=2.0*np.pi*(day-1)/365.0
    rho2=1.00011+0.034221*np.cos(dayang_S)+0.00128*np.sin(dayang_S)+0.000719*np.cos(2*dayang_S)+0.000077*np.sin(2*dayang_S)
    # declination (Bourges 1985)
    J0=78.801+0.2422*(year-1969)-(0.25*(year-1969))
    Jday=day+time/24.0
    w=(360.0/365.25)
    t=Jday-0.5-J0
    dayang=w*t
    dayang=dayang*radeg
    delta=0.3723+23.2567*np.sin(dayang)-0.758*np.cos(dayang)+0.1149*np.sin(2*dayang)+0.3656*np.cos(2*dayang)-0.1712*np.sin(3*dayang)+0.0201*np.cos(3*dayang)
    declination=delta
    delta=delta*radeg
    #Equation of time (spencer,1971)
    daynumberS=2*np.pi*(Jday-1)/365.0
    Eqtime=0.000075+0.001868*np.cos(daynumberS)-0.032077*np.sin(daynumberS)-0.014615*np.cos(2*daynumberS)-0.040849*np.sin(2*daynumberS)
    Eqtime=Eqtime*12.0/np.pi
    # hour angle (omega)
    stnmeridian=timezone*15
    deltalattime=longitude-stnmeridian
    deltalattime=deltalattime*24.0/360.0
    omega=np.pi*(((time+deltalattime+Eqtime)/12.0)-1.0)
    lat=latitude*radeg
    #sunvector(Corripio, 2001)
    sunvector=[0.,0.,0.]
    sunvector[0]=-np.sin(omega)*np.cos(delta)
    sunvector[1]=np.sin(lat)*np.cos(omega)*np.cos(delta)-np.cos(lat)*np.sin(delta)
    sunvector[2]=np.cos(lat)*np.cos(omega)*np.cos(delta)+np.sin(lat)*np.sin(delta)
    #solar elevation
    zenith=np.arccos(sunvector[2])/radeg
    #solar azimuth
    if (sunvector[0]==0)and(sunvector[1]==0):
        azimuth=0.0
    else:
        azimuth=((np.pi)-atan2(sunvector[0],sunvector[1]))/radeg
    ##sunrise and sunset
    omeganul=np.arccos(-np.tan(lat)*np.tan(delta))
    if (limb==1):
        omeganul=omeganul+((0.5*100.0/60.0)*radeg)
    sunrise=12.0*(1.0-(omeganul)/(np.pi))-deltalattime-Eqtime
    sunset=12.0*(1.0+(omeganul)/(np.pi))-deltalattime-Eqtime
    solpos=[sunvector, rho2, azimuth, zenith ,declination ,sunrise, sunset, Eqtime ,omega]
    return solpos

#lat = 27.69
#lon = 86.87
#timezone = 4.5
#time = [5., 8.,10.,12.,14.,16.]
#year = 1999
#day = 244
#limb = 0

def sza(dt, **kwargs):
    """
    solar zenital angle calculator
    params:
    -------
        dt: datetime
    keyword arguments:
    -----------------
        location: some special locations
            mezon = my garden
            osug-b
            gda = galerie de l'alpe
            fluxalp = pré des charmasses
        GPS: (lon,lat)
        timezone: some useful timezones
            utc
            hiver (utc+1)
            ete (utc+2)
    """
    location = 'fluxalp'
    if 'location' in kwargs:
        location = kwargs['location']

    GPS_coord = {'mezon' : (45.1993, 5.8334),
                 'osug-b' : (45.194111, 5.762373),
                 'gda' : (45.035202, 6.400950), 
                 'fluxalp' : (45.041074, 6.410870),
                 'lautaret' : (45.035202, 6.400950)}

    try:
        dt = pd.Timestamp(dt)
    except NameError:
        from pandas import Timestamp
        dt = Timestamp(dt)
        
    if 'GPS' in kwargs:
        GPS = kwargs['GPS']
    else:
        GPS = GPS_coord[location]

    timezones = {'utc': 0,
                 'hiver': 1,
                 'ete': 2}
    if 'timezone' in kwargs: 
        tz = kwargs['timezone']
    else:
        if dt.tz:
            tz = str(dt)[-6:-3]
        else:
            tz = 0
    try:
        tz = int(tz)
    except ValueError:
        tz = timezones[tz]

    day = dt.dayofyear
    time = dt.hour + dt.minute/60 + dt.second/3600
    return solarposnd(GPS[0], GPS[1], day, time, dt.year, tz,0)[3]

if __name__ == '__main__':
    lat=45.04
    lon=6.41
    timezone=1.0
    year=2013
    day=72
    limb=0
    time=[11.]

    for i in range(len(time)):
        a= solarposnd(lat,lon,day,time[i],year,timezone,limb)
        print( a[5], a[6], a[3], cos(a[3]*pi/180.) )
