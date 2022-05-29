import scipy.io as scio
import numpy as np
import pickle
import random

transmitters = {
    'BPSK', 'CPFSK', 'PSK8', 'PAM4', 'QAM16', 'QAM64', 'QPSK',
    'AM_DSB', 'AM_SSB', 'FM'
    }

# data without noise

path_AM_DSB = 'single_AM_DSB.mat'
data_AM_DSB = scio.loadmat(path_AM_DSB)
b_AM_DSB = list(data_AM_DSB.values())
AM_DSB=b_AM_DSB[3]

path_AM_SSB = 'single_AM_SSB.mat'
data_AM_SSB = scio.loadmat(path_AM_SSB)
b_AM_SSB = list(data_AM_SSB.values())
AM_SSB=b_AM_SSB[3]

path_BPSK = 'single_BPSK.mat'
data_BPSK = scio.loadmat(path_BPSK)
b_BPSK = list(data_BPSK.values())
BPSK=b_BPSK[3]

path_CPFSK = 'single_CPFSK.mat'
data_CPFSK = scio.loadmat(path_CPFSK)
b_CPFSK = list(data_CPFSK.values())
CPFSK=b_CPFSK[3]

path_FM = 'single_FM.mat'
data_FM = scio.loadmat(path_FM)
b_FM = list(data_FM.values())
FM=b_FM[3]

# path_GFSK = 'single_GFSK.mat'
# data_GFSK = scio.loadmat(path_GFSK)
# b_GFSK = list(data_GFSK.values())
# GFSK=b_GFSK[3]

path_my8PSK = 'single_my8PSK.mat'
data_my8PSK = scio.loadmat(path_my8PSK)
b_my8PSK = list(data_my8PSK.values())
PSK8=b_my8PSK[3]

path_PAM4 = 'single_PAM4.mat'
data_PAM4 = scio.loadmat(path_PAM4)
b_PAM4 = list(data_PAM4.values())
PAM4=b_PAM4[3]

path_QAM16 = 'single_QAM16.mat'
data_QAM16 = scio.loadmat(path_QAM16)
b_QAM16 = list(data_QAM16.values())
QAM16=b_QAM16[3]

path_QAM64 = 'single_QAM64.mat'
data_QAM64 = scio.loadmat(path_QAM64)
b_QAM64 = list(data_QAM64.values())
QAM64=b_QAM64[3]

path_QPSK = 'single_QPSK.mat'
data_QPSK = scio.loadmat(path_QPSK)
b_QPSK = list(data_QPSK.values())
QPSK=b_QPSK[3]

dataset = {}
# transmitters ={"discrete":['PAM', 'QAM']} # no use
per_num = 1000
single_l = 256
snr_vals = range(-20,20,2)


for snr in snr_vals:
    print(snr)

    for i, mod_type in enumerate(transmitters):
        dataset[(mod_type, snr)] = np.zeros([per_num, 2, single_l], dtype=np.float32)

        for i in range(0, per_num, 1):
            matsnr = int((snr + 20) / 2)
            mod_l = locals()[mod_type]
            dataset[(mod_type, snr)][i, 0, :] = np.real(mod_l[i,matsnr,:])
            dataset[(mod_type, snr)][i, 1, :] = np.imag(mod_l[i,matsnr,:])
        print(mod_type + '_ok!')



print("writing to disk without noise")
f = open('./amc_dataset_10class_tp/radioml_10class_rice_x.pkl', 'wb')
pickle.dump(dataset, f)
print("x all done. ")


############################################ rice h ##############################################

path_AM_DSB = 'single_AM_DSB_h.mat'
data_AM_DSB = scio.loadmat(path_AM_DSB)
b_AM_DSB = list(data_AM_DSB.values())
AM_DSB=b_AM_DSB[3]

path_AM_SSB = 'single_AM_SSB_h.mat'
data_AM_SSB = scio.loadmat(path_AM_SSB)
b_AM_SSB = list(data_AM_SSB.values())
AM_SSB=b_AM_SSB[3]

path_BPSK = 'single_BPSK_h.mat'
data_BPSK = scio.loadmat(path_BPSK)
b_BPSK = list(data_BPSK.values())
BPSK=b_BPSK[3]

path_CPFSK = 'single_CPFSK_h.mat'
data_CPFSK = scio.loadmat(path_CPFSK)
b_CPFSK = list(data_CPFSK.values())
CPFSK=b_CPFSK[3]

path_FM = 'single_FM_h.mat'
data_FM = scio.loadmat(path_FM)
b_FM = list(data_FM.values())
FM=b_FM[3]

# path_GFSK = 'single_GFSK_h.mat'
# data_GFSK = scio.loadmat(path_GFSK)
# b_GFSK = list(data_GFSK.values())
# GFSK=b_GFSK[3]

path_my8PSK = 'single_my8PSK_h.mat'
data_my8PSK = scio.loadmat(path_my8PSK)
b_my8PSK = list(data_my8PSK.values())
PSK8=b_my8PSK[3]

path_PAM4 = 'single_PAM4_h.mat'
data_PAM4 = scio.loadmat(path_PAM4)
b_PAM4 = list(data_PAM4.values())
PAM4=b_PAM4[3]

path_QAM16 = 'single_QAM16_h.mat'
data_QAM16 = scio.loadmat(path_QAM16)
b_QAM16 = list(data_QAM16.values())
QAM16=b_QAM16[3]

path_QAM64 = 'single_QAM64_h.mat'
data_QAM64 = scio.loadmat(path_QAM64)
b_QAM64 = list(data_QAM64.values())
QAM64=b_QAM64[3]

path_QPSK = 'single_QPSK_h.mat'
data_QPSK = scio.loadmat(path_QPSK)
b_QPSK = list(data_QPSK.values())
QPSK=b_QPSK[3]

dataset = {}
# transmitters ={"discrete":['PAM', 'QAM']} # no use
per_num = 1000
single_l = 256
snr_vals = range(-20,20,2)


for snr in snr_vals:
    print(snr)

    for i, mod_type in enumerate(transmitters):
        dataset[(mod_type, snr)] = np.zeros([per_num, 2, single_l], dtype=np.float32)

        for i in range(0, per_num, 1):
            matsnr = int((snr + 20) / 2)
            mod_l = locals()[mod_type]
            dataset[(mod_type, snr)][i, 0, :] = np.real(mod_l[i,matsnr,:])
            dataset[(mod_type, snr)][i, 1, :] = np.imag(mod_l[i,matsnr,:])
        print(mod_type + '_ok!')



print("writing to disk with h")
f = open('./amc_dataset_10class_tp/radioml_10class_rice_x_h.pkl', 'wb')
pickle.dump(dataset, f)
print("xh all done. ")


# ###########################################  rice + noise ###########################################

path_AM_DSB = 'single_AM_DSB_hnoise.mat'
data_AM_DSB = scio.loadmat(path_AM_DSB)
b_AM_DSB = list(data_AM_DSB.values())
AM_DSB=b_AM_DSB[3]

path_AM_SSB = 'single_AM_SSB_hnoise.mat'
data_AM_SSB = scio.loadmat(path_AM_SSB)
b_AM_SSB = list(data_AM_SSB.values())
AM_SSB=b_AM_SSB[3]

path_BPSK = 'single_BPSK_hnoise.mat'
data_BPSK = scio.loadmat(path_BPSK)
b_BPSK = list(data_BPSK.values())
BPSK=b_BPSK[3]

path_CPFSK = 'single_CPFSK_hnoise.mat'
data_CPFSK = scio.loadmat(path_CPFSK)
b_CPFSK = list(data_CPFSK.values())
CPFSK=b_CPFSK[3]

path_FM = 'single_FM_hnoise.mat'
data_FM = scio.loadmat(path_FM)
b_FM = list(data_FM.values())
FM=b_FM[3]

# path_GFSK = 'single_GFSK_hnoise.mat'
# data_GFSK = scio.loadmat(path_GFSK)
# b_GFSK = list(data_GFSK.values())
# GFSK=b_GFSK[3]

path_my8PSK = 'single_my8PSK_hnoise.mat'
data_my8PSK = scio.loadmat(path_my8PSK)
b_my8PSK = list(data_my8PSK.values())
PSK8=b_my8PSK[3]

path_PAM4 = 'single_PAM4_hnoise.mat'
data_PAM4 = scio.loadmat(path_PAM4)
b_PAM4 = list(data_PAM4.values())
PAM4=b_PAM4[3]

path_QAM16 = 'single_QAM16_hnoise.mat'
data_QAM16 = scio.loadmat(path_QAM16)
b_QAM16 = list(data_QAM16.values())
QAM16=b_QAM16[3]

path_QAM64 = 'single_QAM64_hnoise.mat'
data_QAM64 = scio.loadmat(path_QAM64)
b_QAM64 = list(data_QAM64.values())
QAM64=b_QAM64[3]

path_QPSK = 'single_QPSK_hnoise.mat'
data_QPSK = scio.loadmat(path_QPSK)
b_QPSK = list(data_QPSK.values())
QPSK=b_QPSK[3]

dataset = {}
# transmitters ={"discrete":['PAM', 'QAM']} # no use
per_num = 1000
single_l = 256
snr_vals = range(-20,20,2)


for snr in snr_vals:
    print(snr)

    for i, mod_type in enumerate(transmitters):
        dataset[(mod_type, snr)] = np.zeros([per_num, 2, single_l], dtype=np.float32)

        for i in range(0, per_num, 1):
            matsnr = int((snr + 20) / 2)
            mod_l = locals()[mod_type]
            dataset[(mod_type, snr)][i, 0, :] = np.real(mod_l[i,matsnr,:])
            dataset[(mod_type, snr)][i, 1, :] = np.imag(mod_l[i,matsnr,:])
        print(mod_type + '_ok!')



print("writing to disk with h noise")
f = open('./amc_dataset_10class_tp/radioml_10class_rice_x_hn.pkl', 'wb')
pickle.dump(dataset, f)
print("xhn all done. ")








