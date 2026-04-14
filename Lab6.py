import cv2
import numpy as np
import matplotlib.pyplot as plt

def horizontal_flip(image):
    return cv2.flip(image, 1)

def random_rotation(image, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_brightness(image, factor=0.2):
    delta = np.random.uniform(-factor, factor)
    image_float = image.astype(np.float32) / 255.0
    image_float = np.clip(image_float + delta, 0.0, 1.0)
    return (image_float * 255).astype(np.uint8)

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize(image):
    return image.astype(np.float32) / 255.0


# ---------- Pipeline Bài 1 ----------

def bai1():
    image_path = "anh_can_ho_mat_tien.jpg"

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Bài 1] Không đọc được file '{image_path}', bỏ qua.")
        return

    img = cv2.resize(img, (224, 224))

    n_show = 5
    augmented = []
    for _ in range(n_show):
        aug = horizontal_flip(img)
        aug = random_rotation(aug, max_angle=15)
        aug = random_brightness(aug, factor=0.2)
        gray = to_grayscale(aug)
        aug_norm = normalize(gray)
        augmented.append(aug_norm)

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    fig.suptitle("Bài 1 — Trên: Gốc | Dưới: Augmented (Grayscale, Normalized)", fontsize=13)

    for i in range(n_show):
        axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Gốc {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(augmented[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Aug {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("bai1_result.png", dpi=100)
    plt.show()
    print("[Bài 1] Hoàn thành.")


bai1()

def add_gaussian_noise(image, mean=0, std=15):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def bai2():
    image_path = "anh_o_to.jpg"

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Bài 2] Không đọc được file '{image_path}', bỏ qua.")
        return

    img = cv2.resize(img, (224, 224))

    n_show = 5
    augmented = []
    for _ in range(n_show):
        aug = add_gaussian_noise(img)
        aug = random_brightness(aug, factor=0.15)
        aug = random_rotation(aug, max_angle=10)
        aug_norm = normalize(aug)
        augmented.append(aug_norm)

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    fig.suptitle("Bài 2 — Trên: Gốc | Dưới: Augmented (Normalized)", fontsize=13)

    for i in range(n_show):
        axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Gốc {i+1}")
        axes[0, i].axis("off")

        rgb = np.clip(augmented[i][:, :, ::-1], 0, 1)
        axes[1, i].imshow(rgb)
        axes[1, i].set_title(f"Aug {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("bai2_result.png", dpi=100)
    plt.show()
    print("[Bài 2] Hoàn thành.")


bai2()

def random_crop(image, crop_ratio=0.85):
    h, w = image.shape[:2]
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    top  = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)
    cropped = image[top:top + crop_h, left:left + crop_w]
    return cv2.resize(cropped, (w, h))

def random_zoom(image, zoom_range=0.2):
    scale = np.random.uniform(1.0, 1.0 + zoom_range)
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    zoomed = cv2.resize(image, (new_w, new_h))
    start_y = (new_h - h) // 2
    start_x = (new_w - w) // 2
    return zoomed[start_y:start_y + h, start_x:start_x + w]

def bai3():
    image_path = "trai_cay_nong_san.jpg"

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Bài 3] Không đọc được file '{image_path}', bỏ qua.")
        return

    img = cv2.resize(img, (224, 224))

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Bài 3 — Grid 3×3 ảnh Augmented (Normalized)", fontsize=13)

    for row in range(3):
        for col in range(3):
            aug = horizontal_flip(img) if np.random.rand() > 0.5 else img.copy()
            aug = random_crop(aug, crop_ratio=0.85)
            aug = random_zoom(aug, zoom_range=0.2)
            aug = random_rotation(aug, max_angle=10)
            aug_norm = normalize(aug)
            rgb = np.clip(aug_norm[:, :, ::-1], 0, 1)
            axes[row, col].imshow(rgb)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("bai3_result.png", dpi=100)
    plt.show()
    print("[Bài 3] Hoàn thành.")


bai3()

def bai4():
    image_path = "phong_noi_that.jpg"

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Bài 4] Không đọc được file '{image_path}', bỏ qua.")
        return

    img = cv2.resize(img, (224, 224))

    n_show = 3
    augmented = []
    for _ in range(n_show):
        aug = random_rotation(img, max_angle=15)
        aug = horizontal_flip(aug)
        aug = random_brightness(aug, factor=0.2)
        gray = to_grayscale(aug)
        aug_norm = normalize(gray)
        augmented.append(aug_norm)

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    fig.suptitle("Bài 4 — Trên: Gốc | Dưới: Augmented (Grayscale, Normalized)", fontsize=13)

    for i in range(n_show):
        axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Gốc {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(augmented[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Aug {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("bai4_result.png", dpi=100)
    plt.show()
    print("[Bài 4] Hoàn thành.")


bai4()