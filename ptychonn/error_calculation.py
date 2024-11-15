import numpy as np

from skimage.transform import resize
from sklearn.metrics import mean_squared_error as mse


def calc_error(amps, phs, true_amp, true_ph, point_size, overlap):
    amps = np.array(amps).squeeze()
    phs = np.array(phs).squeeze()

    true_amp = np.array(true_amp).squeeze()
    true_ph = np.array(true_ph).squeeze()
    
    # find the nearest side length which will make amps a grid of 64x64 images
    # this is to arrange the images in a grid and compare with ground truth
    test_side_w = amps.shape[0]
    test_side_h = 1

    true_amp = true_amp.reshape(test_side_h, test_side_w, 64, 64)
    true_ph = true_ph.reshape(test_side_h, test_side_w, 64, 64)

    composite_amp = np.zeros((test_side_h*point_size+overlap,test_side_w*point_size+overlap),float)
    ctr = np.zeros_like(composite_amp)
    data_reshaped = amps.reshape(test_side_h, test_side_w,64,64)[:,:,32-int(overlap/2):32+int(overlap/2),
                                                        32-int(overlap/2):32+int(overlap/2)]

    for i in range(test_side_h):
        for j in range(test_side_w):
            composite_amp[point_size*i:point_size*i+overlap, point_size*j:point_size*j+overlap] += data_reshaped[i,j]
            ctr[point_size*i:point_size*i+overlap, point_size*j:point_size*j+overlap] +=1

    composite_phase = np.zeros((test_side_h*point_size+overlap, test_side_w*point_size+overlap),float)
    ctr = np.zeros_like(composite_phase)
    data_reshaped = phs.reshape(test_side_h,test_side_w,64,64)[:,:,32-int(overlap/2):32+int(overlap/2),
                                                        32-int(overlap/2):32+int(overlap/2)]

    for i in range(test_side_h):
        for j in range(test_side_w):
            composite_phase[point_size*i:point_size*i+overlap, point_size*j:point_size*j+overlap] += data_reshaped[i,j]
            ctr[point_size*i:point_size*i+overlap, point_size*j:point_size*j+overlap] +=1

    stitched_phase = composite_phase[int(overlap/2):-int(overlap/2), int(overlap/2):-int(overlap/2)]/ctr[int(overlap/2)
                                                                        :-int(overlap/2), int(overlap/2):-int(overlap/2)]

    stitched_amp = composite_amp[int(overlap/2):-int(overlap/2), int(overlap/2):-int(overlap/2)]/ctr[int(overlap/2)
                                                                        :-int(overlap/2), int(overlap/2):-int(overlap/2)]

    # print(stitched_amp.shape, stitched_phase.shape)
    stitched_amp_down = resize(stitched_amp, (test_side_h,test_side_w), preserve_range=True, anti_aliasing=True)
    stitched_phase_down = resize(stitched_phase, (test_side_h,test_side_w), preserve_range=True, anti_aliasing=True)

    # true_amp = Y_I_test.reshape(NLTEST,NLTEST,64,64)
    # true_ph = Y_phi_test.reshape(NLTEST,NLTEST,64,64)
    # print(stitched_amp_down.shape, stitched_phase_down.shape,  true_amp[:,:,32,32].shape,  true_ph[:,:,32,32].shape)

    return mse(stitched_amp_down, true_amp[:,:,32,32]), mse(stitched_phase_down, true_ph[:,:,32,32])
