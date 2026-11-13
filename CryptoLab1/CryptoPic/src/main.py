import os
import sys
import json
import csv
import hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
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


def unpad_data(data, block_size=16):
    if not data or len(data) % block_size != 0:
        return data
    pad_len = data[-1]
    if pad_len == 0 or pad_len > block_size:
        return data
    if data[-pad_len:] != bytes([pad_len] * pad_len):
        return data
    return data[:-pad_len]


def calculate_entropy(data):
    if not data:
        return 0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / len(data)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def calculate_correlation(data, width, height):
    img = np.frombuffer(data, dtype=np.uint8).reshape((height, width))

    x_h = img[:, :-1].flatten()
    y_h = img[:, 1:].flatten()
    if np.std(x_h) == 0 or np.std(y_h) == 0:
        corr_h = 0.0
    else:
        corr_h = float(np.corrcoef(x_h, y_h)[0, 1])

    x_v = img[:-1, :].flatten()
    y_v = img[1:, :].flatten()
    if np.std(x_v) == 0 or np.std(y_v) == 0:
        corr_v = 0.0
    else:
        corr_v = float(np.corrcoef(x_v, y_v)[0, 1])

    return corr_h, corr_v


def calculate_npcr_uaci(c1, c2):
    c1_arr = np.frombuffer(c1, dtype=np.uint8)
    c2_arr = np.frombuffer(c2, dtype=np.uint8)

    if len(c1_arr) != len(c2_arr):
        return 0.0, 0.0

    diff = c1_arr != c2_arr
    npcr = float(np.sum(diff) / len(c1_arr) * 100)

    uaci = float(
        np.sum(np.abs(c1_arr.astype(int) - c2_arr.astype(int)))
        / (255 * len(c1_arr)) * 100
    )
    return npcr, uaci


def derive_aes_key(password, salt, iterations=200_000):
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations, dklen=16)


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


def get_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(base_dir, "imgs")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return img_dir, results_dir


def encrypt_image(img_name, password, img_dir, results_dir, rows):
    img_path = os.path.join(img_dir, img_name)
    if not os.path.exists(img_path):
        print(f"[skip] нет файла: {img_path}")
        return

    img = Image.open(img_path).convert('L')
    width, height = img.size
    original_bytes = img.tobytes()
    orig_hash = hashlib.sha256(original_bytes).hexdigest()

    salt = os.urandom(16)
    iterations = 200_000
    aes_key = derive_aes_key(password, salt, iterations)
    aes_nonce = os.urandom(16)

    cipher_xor = xor_crypt(original_bytes, password)
    cipher_ecb = aes_ecb_encrypt(original_bytes, aes_key)
    cipher_ctr = aes_ctr_encrypt(original_bytes, aes_key, aes_nonce)

    stem = os.path.splitext(img_name)[0]
    paths = {
        "xor": os.path.join(results_dir, f"{stem}.xor.bin"),
        "ecb": os.path.join(results_dir, f"{stem}.ecb.bin"),
        "ctr": os.path.join(results_dir, f"{stem}.ctr.bin"),
        "meta": os.path.join(results_dir, f"{stem}.meta.json"),
    }

    with open(paths["xor"], "wb") as f:
        f.write(cipher_xor)
    with open(paths["ecb"], "wb") as f:
        f.write(cipher_ecb)
    with open(paths["ctr"], "wb") as f:
        f.write(cipher_ctr)

    meta = {
        "source": img_name,
        "width": width,
        "height": height,
        "sha256_original": orig_hash,
        "kdf": "pbkdf2-hmac-sha256",
        "salt": salt.hex(),
        "iterations": iterations,
        "nonce": aes_nonce.hex(),
        "files": {
            "xor": os.path.basename(paths["xor"]),
            "ecb": os.path.basename(paths["ecb"]),
            "ctr": os.path.basename(paths["ctr"]),
        },
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=4)

    print(f"\nШифрование ({img_name})")

    dec_xor = xor_crypt(cipher_xor, password)
    dec_ecb = aes_ecb_decrypt(cipher_ecb, aes_key)
    dec_ctr = aes_ctr_decrypt(cipher_ctr, aes_key, aes_nonce)

    print("Обратимость:")
    print(f"XOR восстановлен: {hashlib.sha256(dec_xor).hexdigest() == orig_hash}")
    print(f"AES-ECB восстановлен: {hashlib.sha256(dec_ecb).hexdigest() == orig_hash}")
    print(f"AES-CTR восстановлен: {hashlib.sha256(dec_ctr).hexdigest() == orig_hash}")

    ecb_truncated = cipher_ecb[:len(original_bytes)]
    datasets = {
        "Оригинал": original_bytes,
        "XOR": cipher_xor,
        "AES-ECB": ecb_truncated,
        "AES-CTR": cipher_ctr,
    }

    print("\nСтатистическое сравнение")
    for name, data in datasets.items():
        entropy = calculate_entropy(data)
        corr_h, corr_v = calculate_correlation(data, width, height)
        print(f"{name:10} | Энтропия: {entropy:.4f} | Корр H: {corr_h:7.4f} | Корр V: {corr_v:7.4f}")
        rows.append({
            "image": img_name,
            "method": name,
            "entropy": round(entropy, 4),
            "corr_h": round(corr_h, 4),
            "corr_v": round(corr_v, 4),
            "npcr": "",
            "uaci": "",
        })

    key2 = bytearray(aes_key)
    key2[0] ^= 1
    cipher_ctr_key2 = aes_ctr_encrypt(original_bytes, bytes(key2), aes_nonce)
    npcr, uaci = calculate_npcr_uaci(cipher_ctr, cipher_ctr_key2)
    print("\nИзменение ключа")
    print(f"NPCR: {npcr:.2f}%")
    print(f"UACI: {uaci:.2f}%")

    rows.append({
        "image": img_name,
        "method": "AES-CTR (key+1bit)",
        "entropy": "",
        "corr_h": "",
        "corr_v": "",
        "npcr": round(npcr, 2),
        "uaci": round(uaci, 2),
    })

    plt.figure(figsize=(12, 8))
    for i, (name, data) in enumerate(datasets.items(), 1):
        plt.subplot(2, 2, i)
        plt.hist(np.frombuffer(data, dtype=np.uint8), bins=256, color='teal', alpha=0.7)
        plt.title(f"Распределение: {name}")
        plt.xlabel("Яркость пикселя")
        plt.ylabel("Кол-во")
    plt.tight_layout()
    hist_path = os.path.join(results_dir, f"histograms_{stem}.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Гистограммы: {hist_path}")


def decrypt_image(stem, password, results_dir):
    meta_path = os.path.join(results_dir, f"{stem}.meta.json")
    if not os.path.exists(meta_path):
        print(f"[error] нет meta: {meta_path}")
        return

    with open(meta_path, "r") as f:
        meta = json.load(f)

    salt = bytes.fromhex(meta["salt"])
    iterations = meta["iterations"]
    nonce = bytes.fromhex(meta["nonce"])
    orig_hash = meta["sha256_original"]
    aes_key = derive_aes_key(password, salt, iterations)

    files = meta["files"]
    ok_all = True
    for method in ("xor", "ecb", "ctr"):
        path = os.path.join(results_dir, files[method])
        with open(path, "rb") as f:
            blob = f.read()

        if method == "xor":
            dec = xor_crypt(blob, password)
        elif method == "ecb":
            dec = aes_ecb_decrypt(blob, aes_key)
        else:
            dec = aes_ctr_decrypt(blob, aes_key, nonce)

        ok = hashlib.sha256(dec).hexdigest() == orig_hash
        ok_all = ok_all and ok
        print(f"{method}: {'OK' if ok else 'FAIL'}")

    print(f"[{stem}] полное восстановление: {ok_all}")


def write_metrics_csv(rows, results_dir, filename="metrics.csv"):
    if not rows:
        return
    path = os.path.join(results_dir, filename)
    fieldnames = ["image", "method", "entropy", "corr_h", "corr_v", "npcr", "uaci"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nСводный CSV: {path}")


def main():
    img_dir, results_dir = get_paths()

    if len(sys.argv) >= 2 and sys.argv[1] == "decrypt":
        if len(sys.argv) < 4:
            print("usage: python main.py decrypt <stem> <password>")
            return
        stem = os.path.splitext(sys.argv[2])[0]
        password = sys.argv[3]
        print(f"Расшифрование {stem}:")
        decrypt_image(stem, password, results_dir)
        return

    if len(sys.argv) >= 2 and sys.argv[1] != "decrypt":
        password = sys.argv[1]
    else:
        password = "secret_pass"

    images = ["gradient.png", "checkerboard.png"]

    rows = []
    for img_name in images:
        encrypt_image(img_name, password, img_dir, results_dir, rows)

    write_metrics_csv(rows, results_dir)


if __name__ == "__main__":
    main()