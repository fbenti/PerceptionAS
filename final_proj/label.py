import numpy as np

def getLable_number(output_path):
    lable = []
    # for i in range(0,90):
    #     lable.append('box') 
    for i in range(1,102):
        lable.append('book') 
    for i in range(102,102+114):
        lable.append('mug')             
    print(len(lable))

    np.savetxt(output_path + 'lable.txt', lable, fmt="%s")

getLable_number('./')