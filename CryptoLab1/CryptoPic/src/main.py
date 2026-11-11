import os
import json
import hashlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class XorShift32:
    def __init__(self, seed):
        self.state = seed if seed != 0 else 1

    def next(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

def generate_keystream(password, length):
    h = hashlib.sha256(password.encode()).digest()
    seed = int.from_bytes(h[:4], 'little')
    rng = XorShift32(seed)
    
    keystream = bytearray()
    while len(keystream) < length:
        val = rng.next()
        keystream.extend(val.to_bytes(4, 'little'))
    return bytes(keystream[:length])

def xor_crypt(data, password):
    keystream = generate_keystream(password, len(data))
    return bytes([d ^ k for d, k in zip(data, keystream)])


def pad_data(data, block_size=16):
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)

def unpad_data(data):
    pad_len = data[-1]
    if pad_len > 16 or pad_len == 0:
        return data 
    return data[:-pad_len]

def calculate_entropy(data):
    if not data: return 0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / len(data)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def calculate_correlation(data, width, height):
    img = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
    
    x_h = img[:, :-1].flatten()
    y_h = img[:, 1:].flatten()
    if np.std(x_h) == 0 or np.std(y_h) == 0:
        corr_h = 0.0
    else:
        corr_h = np.corrcoef(x_h, y_h)[0, 1]
    
    x_v = img[:-1, :].flatten()
    y_v = img[1:, :].flatten()
    if np.std(x_v) == 0 or np.std(y_v) == 0:
        corr_v = 0.0
    else:
        corr_v = np.corrcoef(x_v, y_v)[0, 1]
    
    return corr_h, corr_v

def calculate_npcr_uaci(c1, c2):
    c1_arr = np.frombuffer(c1, dtype=np.uint8)
    c2_arr = np.frombuffer(c2, dtype=np.uint8)
    
    if len(c1_arr) != len(c2_arr): return 0, 0
    
    diff = c1_arr != c2_arr
    npcr = np.sum(diff) / len(c1_arr) * 100
    
    uaci = np.sum(np.abs(c1_arr.astype(int) - c2_arr.astype(int))) / (255 * len(c1_arr)) * 100
    return npcr, uaci


def aes_ecb_encrypt(data, key):
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    padded = pad_data(data)
    return encryptor.update(padded) + encryptor.finalize()

def aes_ecb_decrypt(data, key):
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(data) + decryptor.finalize()
    return unpad_data(decrypted)

def aes_ctr_encrypt(data, key, nonce):
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(data) + encryptor.finalize()

def aes_ctr_decrypt(data, key, nonce):
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(data) + decryptor.finalize()


def main():

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(base_dir, "imgs")
    results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(results_dir, exist_ok=True)
    
    img_name = "gradient.png"
    img_path = os.path.join(img_dir, img_name)
    
    xor_password = "secret_xor_pass"
    aes_key = os.urandom(16) 
    aes_nonce = os.urandom(16) 
        
    try:
        img = Image.open(img_path).convert('L') 
    except FileNotFoundError:
        print(f"ошибка")
        return

    width, height = img.size
    original_bytes = img.tobytes()
    
    print(f"Шифрование ({img_name})")
    
    cipher_xor = xor_crypt(original_bytes, xor_password)
    
    cipher_ecb = aes_ecb_encrypt(original_bytes, aes_key)
    
    cipher_ctr = aes_ctr_encrypt(original_bytes, aes_key, aes_nonce)
    
    meta = {
        "algorithm": "aes-ctr",
        "nonce": aes_nonce.hex(),
        "source": img_name
    }
    meta_path = os.path.join(results_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    print("\n Обратимость")
    
    dec_xor = xor_crypt(cipher_xor, xor_password)
    dec_ecb = aes_ecb_decrypt(cipher_ecb, aes_key)
    
    with open(meta_path, "r") as f:
        loaded_meta = json.load(f)
    loaded_nonce = bytes.fromhex(loaded_meta["nonce"])
    dec_ctr = aes_ctr_decrypt(cipher_ctr, aes_key, loaded_nonce)
    
    orig_hash = hashlib.sha256(original_bytes).hexdigest()
    print(f"SHA-256: {orig_hash[:16]}...")
    print(f"XOR восстановлен: {hashlib.sha256(dec_xor).hexdigest() == orig_hash}")
    print(f"AES-ECB восстановлен: {hashlib.sha256(dec_ecb).hexdigest() == orig_hash}")
    print(f"AES-CTR восстановлен: {hashlib.sha256(dec_ctr).hexdigest() == orig_hash}")
    
    print("\nСтатистическое сравнение")
    
    ecb_bytes_for_stats = cipher_ecb[:len(original_bytes)]
    
    datasets = {
        "Оригинал": original_bytes,
        "XOR": cipher_xor,
        "AES-ECB": ecb_bytes_for_stats,
        "AES-CTR": cipher_ctr
    }
    
    for name, data in datasets.items():
        entropy = calculate_entropy(data)
        corr_h, corr_v = calculate_correlation(data, width, height)
        print(f"{name:10} | Энтропия: {entropy:.4f} | Корр H: {corr_h:7.4f} | Корр V: {corr_v:7.4f}")

    print("\nИзменение ключа")
    
    key2 = bytearray(aes_key)
    key2[0] ^= 1 
    key2 = bytes(key2)
    
    cipher_ctr_key2 = aes_ctr_encrypt(original_bytes, key2, aes_nonce)
    
    npcr, uaci = calculate_npcr_uaci(cipher_ctr, cipher_ctr_key2)
    print(f"NPCR: {npcr:.2f}% ")
    print(f"UACI: {uaci:.2f}% ")

    plt.figure(figsize=(12, 8))
    for i, (name, data) in enumerate(datasets.items(), 1):
        plt.subplot(2, 2, i)
        plt.hist(np.frombuffer(data, dtype=np.uint8), bins=256, color='teal', alpha=0.7)
        plt.title(f"Распределение: {name}")
        plt.xlabel("Яркость пикселя")
        plt.ylabel("Кол-во")
    plt.tight_layout()
    hist_path = os.path.join(results_dir, "histograms.png")
    plt.savefig(hist_path)

if __name__ == "__main__":
    main()