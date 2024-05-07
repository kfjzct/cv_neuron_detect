from skimage.filters import gaussian, threshold_otsu
from skimage.color import rgb2gray, label2rgb
from skimage.morphology import dilation, disk
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.cluster import DBSCAN
import scipy.ndimage as ndi


def background_correction(img, sigma=5):
    """ Apply Gaussian blur for background noise reduction. """
    return gaussian(img, sigma=sigma)


def binarize_image(image):
    """ Binarize an image based on Otsu's thresholding after subsampling. """
    subsampled_image = subsample(image)
    thresh = threshold_otsu(subsampled_image)
    binary = image > thresh
    return dilation(binary, disk(10))


def subsample(image, reduction_factor=3):
    """ Subsample the image to reduce processing time. """
    x, y = image.shape
    crop_size_x = x // reduction_factor
    crop_size_y = y // reduction_factor
    start_x = (x // 2) - (crop_size_x // 2)
    start_y = (y // 2) - (crop_size_y // 2)
    return image[start_x:start_x + crop_size_x, start_y:start_y + crop_size_y]


def wide_clusters(img, sigma, pixel_density, min_samples, plot=True):
    """ Detect and process wide clusters in an image. """
    grayscale = rgb2gray(img)
    gauss = gaussian(grayscale, sigma=sigma)
    img_subsampled = subsample(gauss)
    thresh = threshold_otsu(img_subsampled)

    is_peak = peak_local_max(gauss, min_distance=int(2.5 * pixel_density),
                             threshold_abs=thresh + (10 * thresh) / 100,
                             exclude_border=False)
    labels, num_features = ndi.label(is_peak)
    local_maxi = np.array(ndi.center_of_mass(is_peak, labels, range(1, num_features + 1)))

    db = DBSCAN(eps=20.6 * pixel_density, min_samples=min_samples).fit(local_maxi)
    labels = db.labels_

    if plot:
        visualize_clusters(gauss, local_maxi, labels)

    return local_maxi, labels, gauss


def visualize_clusters(gauss, local_maxi, labels):
    """ Visualize clustering results. """
    label_plot = np.copy(labels)
    label_plot[labels == 0] = max(labels) + 1
    label_plot[labels == -1] = 0

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), sharex=True, sharey=True)
    ax[0].imshow(gauss, alpha=0.6)
    ax[0].scatter(local_maxi[:, 1], local_maxi[:, 0], s=4)
    ax[0].axis("off")

    ax[1].imshow(gauss, alpha=0.3)
    ax[1].scatter(local_maxi[:, 1], local_maxi[:, 0], c=label_plot, cmap="nipy_spectral")
    ax[1].axis("off")
    plt.show()


def segmentation(img, local_maxi, labels, meta, directory, plot=True, save=False):
    """ Segment the image using watershed algorithm and display/save results if required. """
    only_clusters = np.zeros(img.shape, dtype=int)
    for pos, label in zip(local_maxi, labels):
        if label > 0:
            only_clusters[int(pos[0]), int(pos[1])] = label
        elif label == 0:
            only_clusters[int(pos[0]), int(pos[1])] = max(labels) + 1
    only_clusters = dilation(only_clusters, disk(10))

    binary = binarize_image(img)
    dist_water = ndi.distance_transform_edt(binary)
    segmentation_ws = watershed(-img, only_clusters, mask=binary)

    ganglion_prop = regionprops(segmentation_ws)

    if plot:
        visualize_segmentation(img, segmentation_ws, ganglion_prop, directory, meta, save)

    return ganglion_prop


def visualize_segmentation(img, segmentation_ws, ganglion_prop, directory, meta, save):
    """ Visualize and save segmentation overlay. """
    image_label_overlay = label2rgb(segmentation_ws, image=img.astype('uint16'), bg_label=0)

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.imshow(image_label_overlay, interpolation='nearest')
    ax.axis('off')

    for prop in ganglion_prop:
        ax.annotate(prop.label, (prop.centroid[1] - 5, prop.centroid[0]), color='green', fontsize=8, weight="bold")

    if save:
        filename = os.path.join(directory, f"{meta['Name']}.pdf")
        try:
            plt.savefig(filename, transparent=True)
        except IOError:
            plt.savefig(meta['Name'] + '.pdf', transparent=True)

    plt.show()
