import os
orig_path = "/data/xjtupanda/my_dataset/ME_dataset/orig_data/casme/CAS(ME)2_longVideoFaceCropped/longVideoFaceCropped/s27/27_0505funnyinnovations"
target_path = "/data/xjtupanda/my_dataset/ME_dataset/orig_data/optical_flow/casme/s27"
cmd = (f'denseflow "{orig_path}" -v -b=10 -a=tvl1 '
                               f'-s={1} -if -o="{target_path}"')
os.system(cmd)
