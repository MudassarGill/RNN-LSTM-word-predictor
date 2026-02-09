import scipy.io
import os

try:
    signal_data = scipy.io.loadmat('data/154100_signal.mat')
    print("Signal keys:", [k for k in signal_data.keys() if not k.startswith('__')])
    for k, v in signal_data.items():
        if not k.startswith('__'):
            print(f"  {k}: {v.shape} {v.dtype}")
            
    bp_data = scipy.io.loadmat('data/154100_bp.mat')
    print("BP keys:", [k for k in bp_data.keys() if not k.startswith('__')])
    for k, v in bp_data.items():
        if not k.startswith('__'):
            print(f"  {k}: {v.shape} {v.dtype}")
except Exception as e:
    print(f"Error: {e}")
