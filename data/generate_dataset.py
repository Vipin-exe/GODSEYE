import os
import numpy as np
import pandas as pd

def generate_synthetic_dataset(output_path='data/malware_dataset.csv', n_samples=5000):
    """
    Generates a realistic synthetic dataset mimicking malware and benign software features.
    Features include opcode frequencies, API call counts, and permission flags.
    Simulates class imbalance (e.g., 85% benign, 15% malware).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 0 = Benign, 1 = Malware. Introducing class imbalance.
    # We use 85% benign, 15% malware to match real-world scenarios and test SMOTE.
    labels = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    
    data = []
    for label in labels:
        if label == 0:
            # Benign software characteristics
            # Moderate to low opcode usage of specific types
            mov_freq = np.random.normal(500, 100)
            push_freq = np.random.normal(200, 50)
            jmp_freq = np.random.normal(100, 30)
            
            # API Calls (Safe ones more common, risky ones less common)
            create_file = np.random.poisson(10)
            virtual_alloc = np.random.poisson(2)
            create_remote_thread = np.random.poisson(0.1) # Rare in benign
            internet_connect = np.random.poisson(5)
            
            # Permissions (e.g., Android style or generic system flags)
            has_network_access = np.random.choice([0, 1], p=[0.2, 0.8])
            has_admin_rights = np.random.choice([0, 1], p=[0.9, 0.1])
            has_startup_reg = np.random.choice([0, 1], p=[0.8, 0.2])
            
        else:
            # Malware characteristics
            # High usage of certain opcodes due to packing/obfuscation
            mov_freq = np.random.normal(1500, 300)
            push_freq = np.random.normal(800, 200)
            jmp_freq = np.random.normal(400, 100)
            
            # API Calls (Risky ones more common)
            create_file = np.random.poisson(20)
            virtual_alloc = np.random.poisson(15) # Often used for injection
            create_remote_thread = np.random.poisson(5) # Common in malware
            internet_connect = np.random.poisson(25) # C2 communication
            
            # Permissions
            has_network_access = np.random.choice([0, 1], p=[0.05, 0.95])
            has_admin_rights = np.random.choice([0, 1], p=[0.3, 0.7])
            has_startup_reg = np.random.choice([0, 1], p=[0.1, 0.9])
            
        # Ensure no negative values from normal distribution
        mov_freq = max(0, mov_freq)
        push_freq = max(0, push_freq)
        jmp_freq = max(0, jmp_freq)
        
        data.append([
            mov_freq, push_freq, jmp_freq,
            create_file, virtual_alloc, create_remote_thread, internet_connect,
            has_network_access, has_admin_rights, has_startup_reg,
            label
        ])
        
    columns = [
        'opcode_mov', 'opcode_push', 'opcode_jmp',
        'api_CreateFile', 'api_VirtualAlloc', 'api_CreateRemoteThread', 'api_InternetConnect',
        'perm_NetworkAccess', 'perm_AdminRights', 'perm_StartupRegistry',
        'label'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"[DATA] Generated synthetic dataset with {n_samples} samples at {output_path}")
    print(f"[DATA] Class distribution:\n{df['label'].value_counts(normalize=True) * 100}")
    
    return output_path

if __name__ == "__main__":
    generate_synthetic_dataset()
