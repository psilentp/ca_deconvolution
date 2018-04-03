import numpy as np

def wiener_deconvolution(signal, kernel, snr):
    "lambd is the SNR"
    from scipy import fft,ifft
    kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel)))) # zero pad the kernel to same length
    H = fft(kernel)
    deconvolved = np.real(ifft(fft(signal)*np.conj(H)/(H*np.conj(H) + snr**2)))
    return deconvolved

def make_single_kernel(times,tauon1,tauoff1):
    kx = np.copy(times)
    kon1 = lambda x:np.exp(((-1*tauon1)/(x)))
    koff1 = lambda x:np.exp((-1*x)/tauoff1)
    k1 = (kon1(kx)*koff1(kx))
    return k1/np.max(k1)

def forward_optimize(resampled_ca,
                     resample_times,
                     impulse_idxs,
                     kernel,
                     kernel_window_s = 2,
                     kernel_gain = 0.05):
    kernel = kernel*kernel_gain
    recon = np.zeros_like(resample_times)
    impulses = np.zeros_like(resample_times)
    resample_rate = resample_times[1]-resample_times[0]
    kernel_window_idx = np.int(kernel_window_s*resample_rate)
    max_idx = len(resample_times)
    end_first_loop = resample_times[-1] - kernel_window_s
    last_imp = np.argwhere(impulse_idxs < end_first_loop)[-1][0]
    for imp_idx,nxt_idx in zip(impulse_idxs[:last_imp],impulse_idxs[1:last_imp+1]):
        imp_window = impulses[imp_idx:imp_idx+kernel_window_idx]
        n1 = np.linalg.norm(recon[imp_idx:nxt_idx]-resampled_ca[imp_idx:nxt_idx])
        recon[imp_idx:imp_idx+kernel_window_idx] += kernel[:kernel_window_idx]
        n2 = np.linalg.norm(recon[imp_idx:nxt_idx] - resampled_ca[imp_idx:nxt_idx])
        if n2>n1:
            recon[imp_idx:imp_idx+kernel_window_idx] -= kernel[:kernel_window_idx]
        else:
            impulses[imp_idx] = 1
    #finish up - bit slower
    for imp_idx,nxt_idx in zip(impulse_idxs[last_imp:-2],impulse_idxs[last_imp+1:-1]):
        kw = max_idx-imp_idx
        imp_window = impulses[imp_idx:imp_idx+kw]
        n1 = np.linalg.norm(recon[imp_idx:nxt_idx]-resampled_ca[imp_idx:nxt_idx])
        recon[imp_idx:imp_idx+kw] += kernel[:kw]
        n2 = np.linalg.norm(recon[imp_idx:nxt_idx] - resampled_ca[imp_idx:nxt_idx])
        if n2>n1:
            recon[imp_idx:imp_idx+kw] -= kernel[:kw]
        else:
            impulses[imp_idx] = 1
    return recon,impulses