import os
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.detection import find_peaks

##for all flux levels in a directory separately detect sources and save detected sources as csv file
dir=r'/Users/igezer/ALLWISE/W3'
path = os.getcwd()
obsdir=r'/Users/igezer/ALLWISE/W3'
#files = ['*.fits']
files = glob.glob(os.path.join(obsdir,'*.fits'))
files = glob.glob(os.path.join(obsdir,'*.fits'))





for file in files:
    filename = os.path.basename(file)
    filename_parts = filename.split('_')

    # Extract the RA and Dec values from the file name
    ra_str = filename_parts[2][2:]  # Remove the 'ra' prefix
    dec_str = filename_parts[3][3:]  # Remove the 'dec' prefix
    
    new_filename = f'ra{ra_str}_dec{dec_str}.fits'
    ra = float(ra_str)
    dec = float(dec_str)

    hdu = fits.open(file)[0]
    obs = hdu.data


    mask = np.zeros(obs.shape, dtype=bool)
    mask[25:38, 0:38] = True
    mask[0:15, 0:38] = True
    mask[0:38, 0:15] = True
    mask[0:38, 25:38] = True
    mean, median, std = sigma_clipped_stats(obs, sigma=2.0)
    #print((mean, median, std))
    daofind = DAOStarFinder(fwhm=4, threshold=2.0 * std)
    sources = daofind(obs - median,  mask=mask)

    if sources is not None:
       # Add ra and dec columns to sources table
        sources['ra'] = ra
        sources['dec'] = dec
        for col in sources.colnames:
            sources[col].info.format = '%.8g'  # for consistent table output
            photfile=file+'.csv'
            sources.write(photfile, format='ascii', overwrite=True)
    if sources is None:
        print('No sources were found')

    if sources is not None:

        # Plot the FITS image with the found sources
        plt.imshow(obs, origin='lower', cmap='gray', vmin=mean - 2 * std, vmax=mean + 10 * std)
        plt.scatter(sources['xcentroid'], sources['ycentroid'], facecolors='none', edgecolors='r', marker='o', s=500)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('FITS Image with Detected Sources')

        # # Add a green circle to represent the masked region
        # mask_circle = rect((x, y), radius_pixels, edgecolor='green', facecolor='none')
        # plt.gca().add_patch(mask_circle)


        # Save the plot as an image file with the same name as the FITS file
        plot_filename = os.path.splitext(file)[0] + '.png'
        plt.savefig(plot_filename)
        print(f"Plot saved to: {plot_filename}")

        plt.close()


import pandas as pd
import os
import glob
path = os.getcwd()
filesson = glob.glob(os.path.join('/Users/igezer/ALLWISE/W3/', '*.csv'))
for f in filesson:
    df7 = pd.read_csv(f, sep='\s+', header=None)
    df7.to_csv(f, index=False, header=None)
#############################################################################################

##for just one fits file detect sources and plot
#hdu2 = fits.open('/home/ilknur/AllWISE/w3_plot/ra75.0364844000001_dec-0.793045500000001.fits')[0]
#wcs = WCS(hdu2.header)
#obs2 = hdu2.data
#wcov=obs2[obs2 > 0.0]
#mediancov=np.median(wcov)
#index = obs2 < 0.6*mediancov
#mask = np.zeros(obs2.shape, dtype=bool)
#mask[index] = True
##mask[570:720, 520:720] = True



#hdu1 = fits.open('/home/ilknur/AllWISE/w3_plot/ra75.0364844000001_dec-0.793045500000001.fits')[0]
#obs1 = hdu1.data
#mean, median, std = sigma_clipped_stats(obs1, sigma=2.0)
#print((mean, median, std))
#daofind = DAOStarFinder(fwhm=5, threshold=median + (2. * std))
#sources = daofind(obs1 - median, mask=mask)
##if len(sources) < 1:
    ##print('No sources were found')
##if col in sources.colnames:
    ##sources[col].info.format = '%.8g'  # for consistent table output
#print(sources[:50])
    ##sources.write('values.csv',format='ascii', overwrite=True)
## print only the first 50 peaks
#t = np.absolute(hdu1.header['CDELT1'])*3600.
#positions = np.transpose(sources['xcentroid'], sources['ycentroid'])
#apertures = CircularAperture(positions, r=10/t)
#norm = ImageNormalize(stretch=SqrtStretch())
#plt.imshow(obs1, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
#apertures.plot(color='blue', lw=1.5, alpha=0.5)
#plt.show()



















#threshold = median + (5. * std)
#tbl = find_peaks(obs, threshold, box_size=11)
#tbl['peak_value'].info.format = '%.8g'  # for consistent table output
#print(tbl[:50])  # print only the first 10 peaks
#positions = np.transpose((tbl['x_peak'], tbl['y_peak']))
#apertures = CircularAperture(positions, r=5.)
#norm = simple_norm(obs, 'sqrt', percent=99.9)
#plt.imshow(obs, cmap='Greys_r', origin='lower', norm=norm, interpolation='nearest')
#apertures.plot(color='#0547f9', lw=1.5)
#plt.xlim(0, obs.shape[1] - 1)
#plt.ylim(0, obs.shape[0] - 1)
#plt.show()




