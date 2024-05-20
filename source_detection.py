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






