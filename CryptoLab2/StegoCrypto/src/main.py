import os
import sys
import json
import csv
import hashlib
import struct
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import chi2

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

def derive_aes_key(password, salt, iterations=200_000):
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations, dklen=16)

def get_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(base_dir, "imgs")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return img_dir, results_dir

def aes_ctr_encrypt(data, key, nonce):
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(data) + encryptor.finalize()

def aes_ctr_decrypt(data, key, nonce):
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(data) + decryptor.finalize()

def encrypt_payload(text, password):
    salt = os.urandom(16)
    nonce = os.urandom(16)
    key = derive_aes_key(password, salt)
    ciphertext = aes_ctr_encrypt(text.encode('utf-8'), key, nonce)
    return salt + nonce + ciphertext

def decrypt_payload(payload, password):
    salt = payload[:16]
    nonce = payload[16:32]
    ciphertext = payload[32:]
    key = derive_aes_key(password, salt)
    return aes_ctr_decrypt(ciphertext, key, nonce).decode('utf-8')

def bits_from_bytes(data_bytes):
    return np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))

def bytes_from_bits(bits):
    pad_len = (8 - len(bits) % 8) % 8
    padded_bits = np.pad(bits, (0, pad_len), 'constant')
    return np.packbits(padded_bits).tobytes()

def get_random_positions(seed, total_pixels, needed_bits):
    rng = XorShift32(seed)
    positions = list(range(total_pixels))
    for i in range(total_pixels - 1, 0, -1):
        j = rng.next() % (i + 1)
        positions[i], positions[j] = positions[j], positions[i]
    return positions[:needed_bits]

def embed_lsb(cover_path, payload_bytes, output_path, random_pos=False, seed=0):
    img = Image.open(cover_path).convert('RGB')
    img_array = np.array(img)
    flat_img = img_array.flatten()
    packet = struct.pack('>I', len(payload_bytes)) + payload_bytes
    bits = bits_from_bytes(packet)
    if len(bits) > flat_img.size:
        raise ValueError("Data too large")
    if random_pos:
        positions = get_random_positions(seed, flat_img.size, len(bits))
    else:
        positions = np.arange(len(bits))
    for bit, pos in zip(bits, positions):
        flat_img[pos] = (flat_img[pos] & 0xFE) | bit
    Image.fromarray(flat_img.reshape(img_array.shape)).save(output_path)

def extract_lsb(stego_path, random_pos=False, seed=0):
    img = Image.open(stego_path).convert('RGB')
    flat_img = np.array(img).flatten()
    if random_pos:
        positions_all = get_random_positions(seed, flat_img.size, flat_img.size)
    else:
        positions_all = np.arange(flat_img.size)
    len_bits = [flat_img[positions_all[i]] & 1 for i in range(32)]
    payload_len = struct.unpack('>I', bytes_from_bits(len_bits))[0]
    if payload_len == 0 or payload_len > flat_img.size:
        raise ValueError("Read error")
    data_bits = [flat_img[positions_all[i]] & 1 for i in range(32, 32 + payload_len * 8)]
    return bytes_from_bits(data_bits)

def calculate_chi2(flat_img_channel):
    counts = np.bincount(flat_img_channel, minlength=256)
    chi2_stat = 0
    for i in range(0, 256, 2):
        expected = (counts[i] + counts[i+1]) / 2
        if expected > 0:
            chi2_stat += ((counts[i] - expected) ** 2) / expected
            chi2_stat += ((counts[i+1] - expected) ** 2) / expected
    p_value = 1 - chi2.cdf(chi2_stat, 127)
    return chi2_stat, p_value

def analyze_image(cover_path, stego_path, method_name):
    cover = np.array(Image.open(cover_path).convert('RGB'))
    stego = np.array(Image.open(stego_path).convert('RGB'))
    psnr = peak_signal_noise_ratio(cover, stego, data_range=255)
    ssim = structural_similarity(cover, stego, data_range=255, channel_axis=2)
    changed = np.sum(cover != stego)
    changed_ratio = changed / cover.size
    green_stego = stego[:, :, 1].flatten()
    p_lsb_1 = np.mean((green_stego & 1) == 1)
    chi2_stat, p_value = calculate_chi2(green_stego)
    return {
        "method": method_name,
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "changed_ratio": round(changed_ratio, 4),
        "p_lsb_1": round(p_lsb_1, 4),
        "chi2": round(chi2_stat, 2),
        "p_value": round(p_value, 4)
    }

def plot_difference(cover_path, stego_path, output_path):
    cover = np.array(Image.open(cover_path).convert('RGB')).astype(int)
    stego = np.array(Image.open(stego_path).convert('RGB')).astype(int)
    diff = np.abs(cover - stego)
    diff_amplified = np.clip(diff * 100, 0, 255).astype(np.uint8)
    Image.fromarray(diff_amplified).save(output_path)

def plot_histogram(cover_path, stego_path, output_path):
    cover = np.array(Image.open(cover_path).convert('RGB'))
    stego = np.array(Image.open(stego_path).convert('RGB'))
    plt.figure(figsize=(10, 5))
    plt.hist(cover[:,:,1].flatten(), bins=256, alpha=0.5, label='Cover (Green)', color='green')
    plt.hist(stego[:,:,1].flatten(), bins=256, alpha=0.5, label='Stego (Green)', color='red')
    plt.title('Сравнительная гистограмма')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def embed_experiment(img_name, password, img_dir, results_dir):
    cover_path = os.path.join(img_dir, img_name)
    if not os.path.exists(cover_path):
        return []
    print(f"\nВстраивание ({img_name})")
    stem = os.path.splitext(img_name)[0]
    secret_text = "Laboratory work in cryptography. Secret message."
    payload_plain = secret_text.encode('utf-8')
    path_plain = os.path.join(results_dir, f"stego_plain_{stem}.png")
    embed_lsb(cover_path, payload_plain, path_plain, random_pos=False)
    payload_enc = encrypt_payload(secret_text, password)
    path_enc = os.path.join(results_dir, f"stego_encrypted_{stem}.png")
    embed_lsb(cover_path, payload_enc, path_enc, random_pos=False)
    seed = int.from_bytes(hashlib.sha256(password.encode()).digest()[:4], 'little')
    path_rand = os.path.join(results_dir, f"stego_random_{stem}.png")
    embed_lsb(cover_path, payload_enc, path_rand, random_pos=True, seed=seed)
    rows = []
    rows.append(analyze_image(cover_path, path_plain, "Plain LSB"))
    rows.append(analyze_image(cover_path, path_enc, "Encrypted LSB"))
    rows.append(analyze_image(cover_path, path_rand, "Random LSB"))
    plot_difference(cover_path, path_rand, os.path.join(results_dir, f"difference_{stem}.png"))
    plot_histogram(cover_path, path_rand, os.path.join(results_dir, f"histogram_{stem}.png"))
    for r in rows:
        r["image"] = img_name
    print("\nСтатистическое сравнение")
    print(f"{'Метод':<15} | {'PSNR':<8} | {'SSIM':<7} | {'Changed':<8} | {'P(LSB=1)':<8} | {'Chi2':<10} | {'p-value':<8}")
    for r in rows:
        print(f"{r['method']:<15} | {r['psnr']:<8} | {r['ssim']:<7} | {r['changed_ratio']:<8} | {r['p_lsb_1']:<8} | {r['chi2']:<10} | {r['p_value']:<8}")
    return rows

def payload_volume_experiment(img_name, password, img_dir, results_dir):
    cover_path = os.path.join(img_dir, img_name)
    if not os.path.exists(cover_path):
        return []
    print(f"\nВлияние объёма данных ({img_name})")
    img = Image.open(cover_path).convert('RGB')
    total_bits = img.size[0] * img.size[1] * 3
    stem = os.path.splitext(img_name)[0]
    rows = []
    for percent in [1, 5]:
        target_bytes = int(total_bits * (percent / 100) / 8) - 4
        text = "A" * target_bytes
        payload_enc = encrypt_payload(text, password)
        path_rand = os.path.join(results_dir, f"stego_{percent}percent_{stem}.png")
        embed_lsb(cover_path, payload_enc, path_rand, random_pos=True, seed=42)
        r = analyze_image(cover_path, path_rand, f"{percent}% Payload")
        r["image"] = img_name
        rows.append(r)
    print(f"{'Метод':<15} | {'PSNR':<8} | {'SSIM':<7} | {'Changed':<8} | {'P(LSB=1)':<8} | {'Chi2':<10} | {'p-value':<8}")
    for r in rows:
        print(f"{r['method']:<15} | {r['psnr']:<8} | {r['ssim']:<7} | {r['changed_ratio']:<8} | {r['p_lsb_1']:<8} | {r['chi2']:<10} | {r['p_value']:<8}")
    return rows

def extract_experiment(img_name, password, results_dir):
    stem = os.path.splitext(img_name)[0]
    path_rand = os.path.join(results_dir, f"stego_random_{stem}.png")
    if not os.path.exists(path_rand):
        return
    print(f"\nРасшифрование ({img_name})")
    seed = int.from_bytes(hashlib.sha256(password.encode()).digest()[:4], 'little')
    try:
        extracted_payload = extract_lsb(path_rand, random_pos=True, seed=seed)
        decrypted_text = decrypt_payload(extracted_payload, password)
        print(f"Текст восстановлен: '{decrypted_text}'")
    except Exception:
        pass

def write_metrics_csv(rows, results_dir, filename="metrics.csv"):
    if not rows: return
    path = os.path.join(results_dir, filename)
    fieldnames = ["image", "method", "psnr", "ssim", "changed_ratio", "p_lsb_1", "chi2", "p_value"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    img_dir, results_dir = get_paths()
    images = ["gradient.png", "checkerboard.png"]
    password = sys.argv[2] if len(sys.argv) >= 3 else "secret_pass"
    if len(sys.argv) >= 2 and sys.argv[1] == "decrypt":
        for img_name in images:
            extract_experiment(img_name, password, results_dir)
        return
    print(f"Пароль: {'*' * len(password)}")
    all_rows = []
    for img_name in images:
        rows = embed_experiment(img_name, password, img_dir, results_dir)
        all_rows.extend(rows)
    if all_rows:
        print("\nСводная таблица")
        print(f"{'Изобр.':<16} | {'Метод':<15} | {'PSNR':<8} | {'SSIM':<7} | {'Changed':<8} | {'P(LSB=1)':<8} | {'Chi2':<10} | {'p-value':<8}")
        for r in all_rows:
            print(f"{r.get('image',''):<16} | {r['method']:<15} | {r['psnr']:<8} | {r['ssim']:<7} | {r['changed_ratio']:<8} | {r['p_lsb_1']:<8} | {r['chi2']:<10} | {r['p_value']:<8}")
    volume_rows = payload_volume_experiment(images[0], password, img_dir, results_dir)
    all_rows.extend(volume_rows)
    if all_rows:
        write_metrics_csv(all_rows, results_dir)

if __name__ == "__main__":
    main()