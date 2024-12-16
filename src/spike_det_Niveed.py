import numpy as np
from astropy.io import fits
from tqdm import tqdm
import matplotlib.pyplot as plt

def spike_det():
    file = 'NB3_img.fits'
    with fits.open(file) as hdul:
        img = hdul[0].data

    data0 = img

    sz = data0.shape
    data1 = np.zeros((sz[0] + 20, sz[1] + 20))
    data1[10:10 + sz[0], 10:10 + sz[1]] = data0
    data1[data1 < 0] = 0

    x3 = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    y3 = np.array([1, 1, 1, 0, 0, -1, -1, -1])
    spk_img = np.zeros(sz)
    spk_img0 = np.zeros((sz[0] + 20, sz[1] + 20))
    peri = np.zeros(8)

    th1 = 300  # min intensity threshold
    th2 = 1.2  # median threshold as in AIA image

    for i in tqdm(range(10, 10 + sz[0]), desc='spk detection (%)'):
        for j in range(10, 10 + sz[1]):
            int_val = data1[i, j]
            if int_val > 0:
                im = data1[x3 + i, y3 + j]
                im_gt_0 = im[im > 0]
                if len(im_gt_0) > 0:
                    av = np.mean(im_gt_0)
                    sgma=np.std(im_gt_0)
                    if  int_val>400 and int_val > av +3*sgma:
                        spk_img[i - 10, j - 10] = 1.
                        spk_img0[i, j] = 1.                
                        mn = np.median(im_gt_0)
                        data0[i - 10, j - 10] = mn

    num = np.sum(spk_img == 1)
    print('Number of detected spike =', num)

    np.save('spike_location.npy', spk_img)

    # -------spike replacement
    sz = data0.shape
    data2 = np.zeros((sz[0] + 20, sz[1] + 20))
    data2[10:10 + sz[0], 10:10 + sz[1]] = data0

    c2 = np.where(spk_img0 == 1)
    data2[c2] = -200

    for i in range(sz[0] + 20):
        for j in range(sz[1] + 20):
            if spk_img0[i, j] == 1:
                im = data2[x3 + i, y3 + j] #box around the spike point
                im = im[im != -200]  # Ignore -200 values in the average
                if len(im) > 0:
                    int_val = np.median(im)
                    data2[i, j] = int_val

    data3 = data2[10:10 + sz[0], 10:10 + sz[1]]
    spk_rm_img = data3

    np.save('spike_rm_data.npy', spk_rm_img)

if __name__ == "__main__":
    spike_det()
