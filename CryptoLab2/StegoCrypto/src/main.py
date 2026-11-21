import argparse
import os
import csv
import hashlib
import struct
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import chi2


MESSAGE = "Laboratory work in cryptography. Secret message."
LENGTH_FORMAT = ">I"
LENGTH_BYTES = struct.calcsize(LENGTH_FORMAT)
SALT_BYTES = 16
NONCE_BYTES = 12
TAG_BYTES = 16
ENCRYPTED_OVERHEAD = SALT_BYTES + NONCE_BYTES + TAG_BYTES
METHODS = (
    ("Plain LSB", "plain"),
    ("Encrypted LSB", "encrypted"),
    ("Encrypted Random LSB", "random"),
)


class XorShift32:
    def __init__(self, seed):
        self.state = seed if seed != 0 else 1

    def next(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state


def get_seed(password):
    h = hashlib.sha256(password.encode()).digest()
    return int.from_bytes(h[:4], "little")


def get_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(base_dir, "imgs")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return img_dir, results_dir


def get_random_positions(seed, total, needed):
    rng = XorShift32(seed)
    positions = list(range(total))

    for i in range(total - 1, 0, -1):
        j = rng.next() % (i + 1)
        positions[i], positions[j] = positions[j], positions[i]

    return positions[:needed]


def get_key(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, 200_000, dklen=32
    )


def encrypt_payload(text, password):
    salt = os.urandom(SALT_BYTES)
    nonce = os.urandom(NONCE_BYTES)
    key = get_key(password, salt)
    encrypted = AESGCM(key).encrypt(nonce, text.encode("utf-8"), None)
    return salt + nonce + encrypted


def decrypt_payload(payload, password):
    if len(payload) < ENCRYPTED_OVERHEAD:
        raise ValueError("Слишком короткий зашифрованный пакет")

    nonce_end = SALT_BYTES + NONCE_BYTES
    salt = payload[:SALT_BYTES]
    nonce = payload[SALT_BYTES:nonce_end]
    encrypted = payload[nonce_end:]
    key = get_key(password, salt)

    text = AESGCM(key).decrypt(nonce, encrypted, None)
    return text.decode("utf-8")


def bits_from_bytes(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bytes_from_bits(bits):
    return np.packbits(np.array(bits, dtype=np.uint8)).tobytes()


def embed_lsb(cover_path, payload, output_path, random_pos=False, seed=0):
    with Image.open(cover_path) as img:
        array = np.array(img.convert("RGB"))
    flat = array.reshape(-1)

    if not payload or (LENGTH_BYTES + len(payload)) * 8 > flat.size:
        raise ValueError(f"{cover_path} — пустой пакет или недостаточно места")

    packet = struct.pack(LENGTH_FORMAT, len(payload)) + payload
    bits = bits_from_bytes(packet)
    
    if random_pos:
        positions = get_random_positions(seed, flat.size, bits.size)
    else:
        positions = slice(0, bits.size)

    flat[positions] = (flat[positions] & 0xFE) | bits
    Image.fromarray(array).save(output_path)


def extract_lsb(stego_path, random_pos=False, seed=0):
    with Image.open(stego_path) as img:
        flat = np.array(img.convert("RGB")).reshape(-1)

    header_bits = LENGTH_BYTES * 8
    if flat.size < header_bits:
        raise ValueError("Изображение слишком маленькое для заголовка")

    if random_pos:
        positions = get_random_positions(seed, flat.size, flat.size)
        flat = flat[positions]

    length_bits = flat[:header_bits] & 1
    payload_len = struct.unpack(LENGTH_FORMAT, bytes_from_bits(length_bits))[0]
    needed = header_bits + payload_len * 8

    if payload_len == 0 or needed > flat.size:
        raise ValueError("Неверная длина сообщения")

    data_bits = flat[header_bits:needed] & 1
    return bytes_from_bits(data_bits)


def calculate_chi2(flat):
    counts = np.bincount(flat, minlength=256)
    value = 0
    df = 0

    for i in range(0, 256, 2):
        total = counts[i] + counts[i + 1]
        if total > 0:
            expected = total / 2
            value += (counts[i] - expected) ** 2 / expected
            value += (counts[i + 1] - expected) ** 2 / expected
            df += 1

    p_value = chi2.sf(value, df)
    return value, p_value


def analyze_image(cover_path, stego_path, method):
    with Image.open(cover_path) as img:
        cover = np.array(img.convert("RGB"))
    with Image.open(stego_path) as img:
        stego = np.array(img.convert("RGB"))

    psnr = peak_signal_noise_ratio(cover, stego, data_range=255)
    ssim = structural_similarity(
        cover, stego, data_range=255, channel_axis=2
    )

    changed = np.count_nonzero(cover != stego)
    changed_ratio = changed / cover.size

    cover_flat = cover.reshape(-1)
    flat = stego.reshape(-1)
    cover_p_lsb_1 = np.mean((cover_flat & 1) == 1)
    p_lsb_1 = np.mean((flat & 1) == 1)
    cover_chi2, cover_p_value = calculate_chi2(cover_flat)
    chi2_value, p_value = calculate_chi2(flat)

    return {
        "method": method,
        "psnr": round(float(psnr), 2),
        "ssim": round(float(ssim), 4),
        "changed_components": int(changed),
        "changed_ratio": round(float(changed_ratio), 6),
        "cover_p_lsb_0": round(float(1 - cover_p_lsb_1), 4),
        "cover_p_lsb_1": round(float(cover_p_lsb_1), 4),
        "p_lsb_0": round(float(1 - p_lsb_1), 4),
        "p_lsb_1": round(float(p_lsb_1), 4),
        "cover_chi2": round(float(cover_chi2), 2),
        "cover_p_value": float(cover_p_value),
        "chi2": round(float(chi2_value), 2),
        "p_value": float(p_value)
    }


def plot_difference(cover_path, stego_path, output_path):
    cover = np.array(Image.open(cover_path).convert("RGB")).astype(int)
    stego = np.array(Image.open(stego_path).convert("RGB")).astype(int)
    diff = np.abs(cover - stego)
    diff = np.clip(diff * 100, 0, 255).astype(np.uint8)
    Image.fromarray(diff).save(output_path)


def plot_histogram(cover_path, stego_path, output_path):
    cover = np.array(Image.open(cover_path).convert("RGB")).flatten()
    stego = np.array(Image.open(stego_path).convert("RGB")).flatten()

    plt.figure(figsize=(10, 5))
    plt.hist(cover, bins=256, range=(0, 256), alpha=0.5, label="Cover")
    plt.hist(stego, bins=256, range=(0, 256), alpha=0.5, label="Stego")
    plt.title("Сравнительная гистограмма RGB-компонент")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_metrics(rows):
    print("Метод | Заполнено, % | PSNR | SSIM | Changed | P(LSB=1) | Chi2 | p-value")
    for row in rows:
        print(
            f"{row['method']} | {row['actual_fill_percent']} | {row['psnr']} | "
            f"{row['ssim']} | {row['changed_ratio']} | {row['p_lsb_1']} | "
            f"{row['chi2']} | {row['p_value']:.6g}"
        )


def experiment_text(capacity_bits, percent):
    if percent is None:
        return MESSAGE

    # В процент заполнения входят заголовок и весь пакет AES-GCM.
    packet_bytes = capacity_bits * percent // (100 * 8)
    text_bytes = packet_bytes - LENGTH_BYTES - ENCRYPTED_OVERHEAD
    if text_bytes <= 0:
        raise ValueError(f"При заполнении {percent}% недостаточно места для текста")

    phrase = MESSAGE + " "
    repeats = (text_bytes + len(phrase) - 1) // len(phrase)
    return (phrase * repeats)[:text_bytes]


def stego_filename(cover_path, method_id, percent):
    stem = os.path.splitext(os.path.basename(cover_path))[0]
    suffix = "" if percent is None else f"{percent}percent_"
    return f"stego_{method_id}_{suffix}{stem}.png"


def extract_experiment(stego_path, method_id, password, text, expected_payload=None):
    try:
        random_pos = method_id == "random"
        payload = extract_lsb(stego_path, random_pos, get_seed(password))

        if expected_payload is not None and payload != expected_payload:
            raise ValueError("Извлечённый пакет отличается от записанного")

        if method_id == "plain":
            expected = text.encode("utf-8") + bytes(ENCRYPTED_OVERHEAD)
            matches = payload == expected
        else:
            matches = decrypt_payload(payload, password) == text

        if not matches:
            raise ValueError("Восстановленное сообщение отличается от исходного")
    except InvalidTag as error:
        raise ValueError(
            f"{stego_path} — проверка тега AES-GCM не прошла. "
            "Неверный пароль или повреждённый пакет"
        ) from error
    except (OSError, ValueError) as error:
        raise ValueError(f"{stego_path} — {error}") from error


def embed_experiment(cover_path, password, results_dir, percent=None):
    img_name = os.path.basename(cover_path)
    with Image.open(cover_path) as img:
        capacity_bits = img.width * img.height * 3
    text = experiment_text(capacity_bits, percent)
    plain = text.encode("utf-8")

    encrypted = encrypt_payload(text, password)
    # Для сравнения записываем одинаковое число битов во всех трёх вариантах.
    plain_payload = plain + bytes(ENCRYPTED_OVERHEAD)
    seed = get_seed(password)

    description = "короткое сообщение" if percent is None else f"заполнение {percent}%"
    print(f"\n{img_name} — {description}")
    rows = []

    for method, method_id in METHODS:
        filename = stego_filename(cover_path, method_id, percent)
        stego_path = os.path.join(results_dir, filename)
        payload = plain_payload if method_id == "plain" else encrypted

        embed_lsb(cover_path, payload, stego_path, method_id == "random", seed)
        extract_experiment(stego_path, method_id, password, text, payload)

        row = analyze_image(cover_path, stego_path, method)
        row["image"] = img_name
        row["target_fill_percent"] = "" if percent is None else percent
        row["message_bytes"] = len(plain)
        row["payload_bytes"] = len(payload)
        row["embedded_bits"] = (LENGTH_BYTES + len(payload)) * 8
        row["actual_fill_percent"] = round(row["embedded_bits"] / capacity_bits * 100, 6)
        row["verified"] = "ok"
        rows.append(row)

        if method_id == "random" and percent is None:
            stem = os.path.splitext(img_name)[0]
            plot_difference(
                cover_path, stego_path, os.path.join(results_dir, f"difference_{stem}.png")
            )
            plot_histogram(
                cover_path, stego_path, os.path.join(results_dir, f"histogram_{stem}.png")
            )

    print_metrics(rows)
    return rows


def write_metrics_csv(rows, results_dir):
    path = os.path.join(results_dir, "metrics.csv")
    fields = [
        "image", "method", "target_fill_percent", "message_bytes",
        "payload_bytes", "embedded_bits", "actual_fill_percent", "verified",
        "psnr", "ssim", "changed_components", "changed_ratio",
        "cover_p_lsb_0", "cover_p_lsb_1", "p_lsb_0", "p_lsb_1",
        "cover_chi2", "cover_p_value", "chi2", "p_value"
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Лабораторная работа №2 — LSB")
    parser.add_argument("mode", choices=("embed", "decrypt"), nargs="?", default="embed")
    parser.add_argument("password", nargs="?", default="secret_pass")
    parser.add_argument("--levels", type=int, nargs=2, metavar=("P1", "P2"))
    args = parser.parse_args()
    if args.levels and not 1 <= args.levels[0] < args.levels[1] <= 100:
        parser.error("Укажи два возрастающих процента от 1 до 100")

    img_dir, results_dir = get_paths()
    images = ["gradient.png", "checkerboard.png"]
    if os.path.exists(os.path.join(img_dir, "custom.png")):
        images.append("custom.png")

    try:
        if args.mode == "decrypt":
            for img_name in images:
                cover_path = os.path.join(img_dir, img_name)
                with Image.open(cover_path) as img:
                    capacity_bits = img.width * img.height * 3
                for percent in (None, *(args.levels or [])):
                    text = experiment_text(capacity_bits, percent)
                    for method, method_id in METHODS:
                        filename = stego_filename(cover_path, method_id, percent)
                        stego_path = os.path.join(results_dir, filename)
                        extract_experiment(stego_path, method_id, args.password, text)
                        print(f"{filename} — {method}, сообщение восстановлено правильно")
            return

        levels = args.levels or [1, 5]
        rows = []
        with TemporaryDirectory(dir=results_dir) as temp_dir:
            for img_name in images:
                cover_path = os.path.join(img_dir, img_name)
                for percent in (None, *levels):
                    rows.extend(embed_experiment(cover_path, args.password, temp_dir, percent))
            write_metrics_csv(rows, temp_dir)
            for filename in os.listdir(temp_dir):
                if filename != "metrics.csv":
                    os.replace(os.path.join(temp_dir, filename), os.path.join(results_dir, filename))
            os.replace(os.path.join(temp_dir, "metrics.csv"), os.path.join(results_dir, "metrics.csv"))

        print(f"\nПроверено {len(rows)} стегоизображений")
        print(f"Результаты сохранены в {os.path.join(results_dir, 'metrics.csv')}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Ошибка — {error}\n")


if __name__ == "__main__":
    main()
