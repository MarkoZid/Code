import os
import pickle
import numpy as np
# src=r'D:\Monika\Models\ModelsTestingNew3\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor_Renamed_PRVVTORsloj_16filters_BEZ_SKIP_1INC\reg_coef'
src=r'C:\Users\tatja\Desktop\LETNASKOLA\Models\regCoeffs'

def convertpkltotxt(src,dst):
    norm_coeffs_value=[]
    for filename in os.listdir(src):
        print(filename)
        fid1_2 = open(os.path.join(src, filename), 'rb')
        reg_norm_coef_size = pickle.load(fid1_2)
        print(reg_norm_coef_size)
        fid1_2.close()
        norm_coeffs_value.append(reg_norm_coef_size)

    np.savetxt(dst,[norm_coeffs_value],fmt="%f",delimiter=',')


def loadnormcoeffvalues(src):
    norm_coeffs_value = []
    for filename in os.listdir(src):
        print(filename)
        fid1_2 = open(os.path.join(src, filename), 'rb')
        reg_norm_coef_size = pickle.load(fid1_2)
        print(reg_norm_coef_size)
        fid1_2.close()
        norm_coeffs_value.append(reg_norm_coef_size)

    return norm_coeffs_value