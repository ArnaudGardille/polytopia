#!/usr/bin/env python3
"""Détection du point inférieur unique d'un bloc de terrain et recadrage associé."""

import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def detect_bottom_point(image_path: Path) -> Optional[Tuple[int, int]]:
    """
    Détecte uniquement le point inférieur (centre bas) du bloc.

    Retourne:
        (bottom_x, bottom_y) ou None si la détection échoue.
    """
    img = Image.open(image_path)
    img_array = np.array(img)

    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2).astype(np.uint8)
    else:
        gray = img_array

    threshold = 10
    terrain_mask = gray > threshold

    if not np.any(terrain_mask):
        print(f"  ⚠️  Aucun terrain détecté dans {image_path.name}")
        return None

    terrain_y, terrain_x = np.where(terrain_mask)

    if len(terrain_y) == 0:
        print(f"  ⚠️  Aucun pixel de terrain trouvé dans {image_path.name}")
        return None

    bottom_y = int(np.max(terrain_y))
    bottom_candidates = terrain_x[terrain_y == bottom_y]

    if len(bottom_candidates) == 0:
        print(f"  ⚠️  Aucun point inférieur trouvé dans {image_path.name}")
        return None

    center_x = gray.shape[1] // 2
    idx = int(np.argmin(np.abs(bottom_candidates - center_x)))
    bottom_x = int(bottom_candidates[idx])

    return bottom_x, bottom_y


def visualize_detection(image_path: Path, bottom_point: Tuple[int, int], output_path: Path):
    """Crée une image annotée avec le point inférieur en rouge."""
    img = Image.open(image_path)
    img_array = np.array(img)

    vis_img = img_array.copy()
    x, y = bottom_point

    for dy in range(-5, 6):
        for dx in range(-5, 6):
            if dx * dx + dy * dy <= 25:
                px, py = x + dx, y + dy
                if 0 <= px < vis_img.shape[1] and 0 <= py < vis_img.shape[0]:
                    if vis_img.ndim == 3:
                        if vis_img.shape[2] == 4:
                            vis_img[py, px] = [255, 0, 0, 255]
                        else:
                            vis_img[py, px] = [255, 0, 0]
                    else:
                        vis_img[py, px] = 255

    Image.fromarray(vis_img).save(output_path)
    print(f"  ✓ Visualisation sauvegardée: {output_path}")


def save_debug_artifacts(image_path: Path, terrain_dir: Path, bottom_point: Tuple[int, int]) -> Path:
    """Sauvegarde l'original et la détection dans /debug/<image>/."""
    debug_root = terrain_dir / "debug"
    image_debug_dir = debug_root / image_path.stem
    image_debug_dir.mkdir(parents=True, exist_ok=True)

    original_path = image_debug_dir / f"original_{image_path.name}"
    shutil.copy2(image_path, original_path)

    detection_path = image_debug_dir / f"detection_{image_path.name}"
    visualize_detection(image_path, bottom_point, detection_path)

    return image_debug_dir


def test_detection(terrain_dir: Path, test_images: Optional[List[str]] = None):
    """Teste la détection du point inférieur."""
    print("=" * 60)
    print("Détection du point inférieur")
    print("=" * 60)

    if test_images:
        image_files = [terrain_dir / img for img in test_images if (terrain_dir / img).exists()]
    else:
        defaults = ["Grass.png", "Imperius_ground_with_forest.png", "Imperius_ground_with_mountain.png"]
        image_files = [terrain_dir / img for img in defaults if (terrain_dir / img).exists()]

    if not image_files:
        print(f"⚠️  Aucune image trouvée dans {terrain_dir}")
        return

    for img_path in image_files:
        print(f"\n📷 {img_path.name}")
        bottom_point = detect_bottom_point(img_path)
        if bottom_point:
            debug_dir = save_debug_artifacts(img_path, terrain_dir, bottom_point)
            print(f"  ✓ Point inférieur: ({bottom_point[0]:4d}, {bottom_point[1]:4d})")
            print(f"  ✓ Originaux & détection dans {debug_dir.relative_to(terrain_dir)}")
        else:
            print("  ✗ Détection échouée")


def crop_to_bottom(terrain_dir: Path, backup: bool = True):
    """
    Recadre toutes les images sur le point inférieur et les padde avec de la transparence
    pour obtenir une taille commune (sans déformation).
    """
    print("=" * 60)
    print("Recadrage + padding basé sur le point inférieur unique")
    print("=" * 60)

    image_files = [f for f in terrain_dir.glob("*.png") if not f.name.startswith("detection_")]

    if not image_files:
        print(f"⚠️  Aucune image trouvée dans {terrain_dir}")
        return

    if backup:
        backup_dir = terrain_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"📁 Sauvegarde dans: {backup_dir}")
    else:
        backup_dir = None

    records = []
    max_width = 0
    max_height = 0

    # Première passe : détection et collecte des informations nécessaires
    for img_path in image_files:
        bottom_point = detect_bottom_point(img_path)
        if not bottom_point:
            print(f"⚠️  Détection impossible pour {img_path.name}, ignoré")
            continue

        img = Image.open(img_path)
        width, height = img.size
        bottom_x, bottom_y = bottom_point
        crop_height = max(1, bottom_y + 1)

        debug_dir = save_debug_artifacts(img_path, terrain_dir, bottom_point)

        records.append({
            "path": img_path,
            "width": width,
            "crop_height": crop_height,
            "bottom_point": bottom_point,
            "debug_dir": debug_dir
        })

        max_width = max(max_width, width)
        max_height = max(max_height, crop_height)

    if not records:
        print("❌ Aucune image valide à traiter")
        return

    print(f"Taille cible: {max_width} x {max_height}")

    processed = 0
    for record in records:
        img_path = record["path"]
        width = record["width"]
        crop_height = record["crop_height"]
        bottom_x, bottom_y = record["bottom_point"]
        debug_dir = record["debug_dir"]

        img = Image.open(img_path).convert("RGBA")
        cropped = img.crop((0, 0, width, crop_height))

        scale = min(max_width / width, max_height / crop_height)
        if scale < 1.0:
            # Normalement impossible car max_width/height sont des maxima,
            # mais on garde la logique pour éviter toute surprise.
            scale = 1.0
        scaled_width = int(round(width * scale))
        scaled_height = int(round(crop_height * scale))
        if scale != 1.0:
            resized = cropped.resize((scaled_width, scaled_height), Image.LANCZOS)
        else:
            resized = cropped

        canvas = Image.new("RGBA", (max_width, max_height), (0, 0, 0, 0))
        x_offset = (max_width - scaled_width) // 2
        y_offset = max_height - scaled_height  # aligne le point inférieur sur la base
        canvas.paste(resized, (x_offset, y_offset), mask=resized)

        result_path = debug_dir / f"result_{img_path.name}"
        canvas.save(result_path)

        if backup_dir:
            shutil.copy2(img_path, backup_dir / img_path.name)

        canvas.save(img_path)

        print(f"📷 {img_path.name}")
        print(f"  ✓ Croppé à {width}x{crop_height}, échelle {scale:.2f}, padded à {max_width}x{max_height}")
        print(f"  ✓ Point ({bottom_x},{bottom_y}) aligné au bas, résultat: {result_path.relative_to(terrain_dir)}")
        processed += 1

    print("=" * 60)
    print(f"✅ {processed} image(s) recadrée(s) et harmonisée(s)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Détecte le point inférieur d'un bloc de terrain")
    parser.add_argument(
        "--terrain-dir",
        type=str,
        default="frontend/public/icons/terrain",
        help="Dossier contenant les images de terrain"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Teste uniquement la détection"
    )
    parser.add_argument(
        "--test-images",
        type=str,
        nargs="+",
        help="Liste d'images spécifiques à tester"
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Recadre les images en supprimant tout sous le point inférieur"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Ne pas sauvegarder les images originales"
    )

    args = parser.parse_args()
    terrain_dir = Path(args.terrain_dir)

    if not terrain_dir.exists():
        print(f"❌ Le dossier {terrain_dir} n'existe pas")
        return

    if args.crop:
        crop_to_bottom(terrain_dir, backup=not args.no_backup)
    elif args.test or args.test_images:
        test_detection(terrain_dir, args.test_images)
    else:
        print("Mode test par défaut (utilisez --crop pour recadrer).")
        test_detection(terrain_dir)


if __name__ == "__main__":
    main()
