import hashlib
import numpy as np

def extract_features_from_file(file_path):
    """
    SIMULATION / DEMONSTRATION MODE:
    Real feature extraction (Opcodes, APIs) from an executable requires 
    dynamic sandboxing (like Cuckoo) or static disassemblers (like IDA Pro/pefile).
    
    For the purpose of this demonstration, we read the uploaded file, hash it, 
    and use the hash as a seed to generate synthetic features that perfectly match 
    the schema expected by our ML pipeline. 
    """
    # Read the file to generate a hash
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
        
    file_hash = hasher.hexdigest()
    
    # Use the hash as a seed for deterministic "extraction" (same file = same features)
    seed = int(file_hash[:8], 16)
    np.random.seed(seed)
    
    # We simulate extraction. To make the demo interesting, we'll arbitrarily decide
    # if this file looks like "malware" or "benign" based on the seed being even/odd
    is_malware_like = seed % 2 != 0
    
    if is_malware_like:
        features = [
            max(0, np.random.normal(1500, 300)), # opcode_mov
            max(0, np.random.normal(800, 200)),  # opcode_push
            max(0, np.random.normal(400, 100)),  # opcode_jmp
            np.random.poisson(20),               # api_CreateFile
            np.random.poisson(15),               # api_VirtualAlloc
            np.random.poisson(5),                # api_CreateRemoteThread
            np.random.poisson(25),               # api_InternetConnect
            np.random.choice([0, 1], p=[0.05, 0.95]), # perm_NetworkAccess
            np.random.choice([0, 1], p=[0.3, 0.7]),   # perm_AdminRights
            np.random.choice([0, 1], p=[0.1, 0.9])    # perm_StartupRegistry
        ]
    else:
        features = [
            max(0, np.random.normal(500, 100)),  # opcode_mov
            max(0, np.random.normal(200, 50)),   # opcode_push
            max(0, np.random.normal(100, 30)),   # opcode_jmp
            np.random.poisson(10),               # api_CreateFile
            np.random.poisson(2),                # api_VirtualAlloc
            np.random.poisson(0),                # api_CreateRemoteThread
            np.random.poisson(5),                # api_InternetConnect
            np.random.choice([0, 1], p=[0.2, 0.8]),   # perm_NetworkAccess
            np.random.choice([0, 1], p=[0.9, 0.1]),   # perm_AdminRights
            np.random.choice([0, 1], p=[0.8, 0.2])    # perm_StartupRegistry
        ]
        
    feature_names = [
        'opcode_mov', 'opcode_push', 'opcode_jmp',
        'api_CreateFile', 'api_VirtualAlloc', 'api_CreateRemoteThread', 'api_InternetConnect',
        'perm_NetworkAccess', 'perm_AdminRights', 'perm_StartupRegistry'
    ]
    
    # Return as a dictionary for the UI, and as a list for the ML model
    feature_dict = dict(zip(feature_names, features))
    return feature_dict, features
